# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell, nn_ops, math_ops, array_ops, init_ops
from util import _linear, get_logits, VariationalDropout, TriLinearSim


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob

    def build_graph(self, inputs, masks, scope):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        rnn_cell_fw = DropoutWrapper(rnn_cell_fw, input_keep_prob=self.keep_prob)
        rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        rnn_cell_bw = DropoutWrapper(rnn_cell_bw, input_keep_prob=self.keep_prob)

        input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

        # Note: fw_out and bw_out are the hidden states for every timestep. Each is shape (batch_size, seq_len, hidden_size).
        (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, inputs, input_lens, dtype=tf.float32, scope=scope)

        out = tf.concat([fw_out, bw_out], 2)
        out = VariationalDropout(out, self.keep_prob)
        return out


class LSTM_Mapper(object):
    """
    General-purpose module to encode a sequence using a Bi-LSTM.
    It feeds the input through a RNN and returns all the hidden states.
    No dropout is applied for the output
    """
    def __init__(self, hidden_size, drop_in, keep_prob):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.drop_in = drop_in  # Whether to add dropout wrapper for the input

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with tf.variable_scope("LSTM_Mapper"):
            rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
            rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
            if self.drop_in:
                rnn_cell_fw = DropoutWrapper(rnn_cell_fw, input_keep_prob=self.keep_prob)
                rnn_cell_bw = DropoutWrapper(rnn_cell_bw, input_keep_prob=self.keep_prob)

            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep. Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, inputs, input_lens, dtype=tf.float32)
            out = tf.concat([fw_out, bw_out], 2)
            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


class BidafAttention(object):
    """Module for Bi-directional Attention Flow Layer
    Inputs:
        Contextual representations of context and question
    Returns:
        Query-aware context representations - G
    """

    def __init__(self, keep_prob):
        self.keep_prob = keep_prob

    def build_graph(self, H, H_mask, U, U_mask, d):
        """
        Compared to the old implementation, this version greatly improves the memory efficiency
        This version allows much larger batch size, which in turn gives much better runtime and higher quality updates

        Inputs:
            H: [N, context_len, d]
            U: [N, qn_len, d]
            H_mask: [N, context_len]
            U_mask: [N, qn_len]
        Outputs:
            G: [N, context_len, 4 * d]
        """
        # T: context length, J: question length
        T = H.get_shape().as_list()[1]
        J = U.get_shape().as_list()[1]

        H_mask_new = tf.expand_dims(H_mask, axis=-1) # H_mask: (N, T) -> (N, T, 1)
        U_mask_new = tf.expand_dims(U_mask, axis=1) # U_mask: (N, J) -> (N, 1, J)
        HU_mask_new = tf.cast(H_mask_new, tf.bool) & tf.cast(U_mask_new, tf.bool) # (N, T, J)
        assert H_mask_new.get_shape().as_list() == [None, T, 1], "H_mask_new: expected {}, got {}".format([None, T, 1], H_mask_new.get_shape().as_list())
        assert U_mask_new.get_shape().as_list() == [None, 1, J], "U_mask_new: expected {}, got {}".format([None, 1, J], U_mask_new.get_shape().as_list())
        assert HU_mask_new.get_shape().as_list() == [None, T, J], "HU_mask_new: expected {}, got {}".format([None, T, J], HU_mask_new.get_shape().as_list())

        # Contruct the similarity matrix S, dimension: (N, T, J)
        with vs.variable_scope("AttnSimilarity"):
            S = TriLinearSim(H, U)  # (N, T, J)
            assert S.get_shape().as_list() == [None, T, J], "S: expected {}, got {}".format([None, T, J], S.get_shape().as_list())

        # Context-to-query (C2Q) Attention
        with vs.variable_scope("C2Q"):
            _, a_c2q = masked_softmax(S, HU_mask_new, dim=-1)  # Softmax for each context t, (N, T, J)
            assert a_c2q.get_shape().as_list() == [None, T, J], "a_c2q: expected {}, got {}".format([None, T, J], a_c2q.get_shape().as_list())

            attn_c2q = tf.matmul(a_c2q, U)
            assert attn_c2q.get_shape().as_list() == [None, T, d], "attn_c2q: expected {}, got {}".format([None, T, d], attn_c2q.get_shape().as_list())

        # Query-to-context (Q2C) Attention
        with vs.variable_scope("Q2C"):
            b = tf.reduce_max(S, axis=-1) # reduce across the J-dimension -> (N, T)
            assert b.get_shape().as_list() == [None, T], "b: expected {}, got {}".format([None, T], b.get_shape().as_list())

            _, b = masked_softmax(b, H_mask, dim=-1) # (N, T)
            assert b.get_shape().as_list() == [None, T], "b: expected {}, got {}".format([None, T], b.get_shape().as_list())

            b = tf.expand_dims(b, axis=-1) # (N, T, 1)
            assert b.get_shape().as_list() == [None, T, 1], "b: expected {}, got {}".format([None, T, 1], b.get_shape().as_list())

            attn_q2c = tf.reduce_sum(b * H, axis=1) # reduce across the T-dimension: (N, T, d) -> (N, d)
            attn_q2c = tf.tile(tf.expand_dims(attn_q2c, axis=1), [1, T, 1]) # (N, d) -> (N, 1, d) -> (N, T, d)
            assert attn_q2c.get_shape().as_list() == [None, T, d], "attn_q2c: expected {}, got {}".format([None, T, d], attn_q2c.get_shape().as_list())

        # Generate contextual embeddings using both attn_c2q and attn_q2c
        with vs.variable_scope("AttentionConcat"):
            attn_result = tf.concat([H, attn_c2q, H * attn_c2q, H * attn_q2c], -1) # (N, T, 4 * d)
            assert attn_result.get_shape().as_list() == [None, T, 4*d], "attn_result: expected {}, got {}".format([None, T, 4*d], attn_result.get_shape().as_list())

        return attn_result


class SelfAttention(object):
    """
        Turns query-aware context representations into final self-attended feature representations
    """

    def __init__(self, hidden_size, keep_prob):
        self.keep_prob = keep_prob
        self.hidden_size = hidden_size

    def build_graph(self, G, G_mask):
        """
        The original r-net attention mechenism is way too memory/computation hungry
        Here we use scaled dot-product attention to reduce time and space requirement (as in Paper "Attention is All You Need")

        Inputs (query-aware context representations):
            G: [N, context_len, 8 * d]  - d is the hidden size
            G_mask: [N, context_len]
        Returns (self-attended context representations):
            M: [N, context_len, 2 * d]
        """
        d = self.hidden_size    # hidden size
        T = G.get_shape().as_list()[1]  # context length

        mask_bool = tf.cast(G_mask, tf.bool)
        mask2d = tf.expand_dims(mask_bool, axis=1) & tf.expand_dims(mask_bool, axis=2)    #  (N, T, T)
        assert mask2d.get_shape().as_list() == [None, T, T], "mask2d: expected {}, got {}".format([None, T, T], mask2d.get_shape().as_list())

        G_d = VariationalDropout(G, self.keep_prob)

        # Use a fully-connected layer to reduce dimension
        Q = tf.contrib.layers.fully_connected(G_d, num_outputs=d, activation_fn=tf.nn.relu, biases_initializer=None) # (N, T, 8*d) -> (N, T, d)
        assert Q.get_shape().as_list() == [None, T, d], "Q: expected {}, got {}".format([None, T, d], Q.get_shape().as_list())

        # Calculate similarity matrix and apply softmax
        sim = tf.matmul(Q, tf.transpose(Q, [0, 2, 1])) / (d ** 0.5) # Similarity matrix - (N, T, T)
        sim_mask = tf.expand_dims(tf.eye(num_rows=T, dtype=tf.float32), axis=0) # (1, T, T)
        sim -= (1e28) * sim_mask  # Force the word to align with other words
        _, sim = masked_softmax(sim, mask2d, dim=-1)    # Softmax similarity - (N, T, T)
        assert sim.get_shape().as_list() == [None, T, T], "sim: expected {}, got {}".format([None, T, T], sim.get_shape().as_list())

        # Take the weighted average to obtain the self-attention vectors
        SelfAttn = tf.matmul(sim, G)    # (N, T, 2*d)

        # Concatenate with the original feature representation
        SelfAttn = tf.concat([SelfAttn, G, SelfAttn * G], axis=-1) # (N, T, 6*d)
        assert SelfAttn.get_shape().as_list() == [None, T, 6*d], "SelfAttn: expected {}, got {}".format([None, T, 6*d], SelfAttn.get_shape().as_list())

        with tf.variable_scope("AttentionGate"):
            SelfAttn_d = VariationalDropout(SelfAttn, self.keep_prob)
            gate = tf.contrib.layers.fully_connected(SelfAttn_d, num_outputs=6*d, activation_fn=tf.nn.sigmoid, biases_initializer=None)
            assert gate.get_shape().as_list() == [None, T, 6*d], "gate: expected {}, got {}".format([None, T, 6*d], gate.get_shape().as_list())

        SelfAttn_final = SelfAttn * gate    # (N, T, 6*d)
        assert SelfAttn_final.get_shape().as_list() == [None, T, 6*d], "SelfAttn_final: expected {}, got {}".format([None, T, 6*d], SelfAttn_final.get_shape().as_list())

        # No dropout is applied to SelfAttn_final yet
        return SelfAttn_final


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
