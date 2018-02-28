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
from util import _linear, get_logits # After re-implementation of BiDAF, these two are no longer used


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
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

        # Concatenate the forward and backward hidden states
        out = tf.concat([fw_out, bw_out], 2)

        # Apply dropout
        out = tf.nn.dropout(out, self.keep_prob)

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
        Compared to the old implementation, this version slightly improves the runtime

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

        H_new = tf.expand_dims(H, axis=2)   # H: (N, T, d) -> (N, T, 1, d)
        U_new = tf.expand_dims(U, axis=1)   # U: (N, J, d) -> (N, 1, J, d)
        H_mask_new = tf.expand_dims(H_mask, axis=-1) # H_mask: (N, T) -> (N, T, 1)
        U_mask_new = tf.expand_dims(U_mask, axis=1) # U_mask: (N, J) -> (N, 1, J)
        HU_mask_new = tf.cast(H_mask_new, tf.bool) & tf.cast(U_mask_new, tf.bool) # (N, T, J)
        assert H_new.get_shape().as_list() == [None, T, 1, d], "H_new: expected {}, got {}".format([None, T, 1, d], H_new.get_shape().as_list())
        assert U_new.get_shape().as_list() == [None, 1, J, d], "U_new: expected {}, got {}".format([None, 1, J, d], U_new.get_shape().as_list())
        assert H_mask_new.get_shape().as_list() == [None, T, 1], "H_mask_new: expected {}, got {}".format([None, T, 1], H_mask_new.get_shape().as_list())
        assert U_mask_new.get_shape().as_list() == [None, 1, J], "U_mask_new: expected {}, got {}".format([None, 1, J], U_mask_new.get_shape().as_list())
        assert HU_mask_new.get_shape().as_list() == [None, T, J], "HU_mask_new: expected {}, got {}".format([None, T, J], HU_mask_new.get_shape().as_list())

        # Contruct the similarity matrix S, dimension: (N, T, J)
        with vs.variable_scope("AttnSimilarity"):
            S_H = tf.contrib.layers.fully_connected(H_new, num_outputs=1, activation_fn=None, biases_initializer=None)   # (N, T, 1, 1)
            S_U = tf.contrib.layers.fully_connected(U_new, num_outputs=1, activation_fn=None, biases_initializer=None)   # (N, 1, J, 1)
            S_HU = tf.contrib.layers.fully_connected(H_new * U_new, num_outputs=1, activation_fn=None, biases_initializer=None) # (N, T, J, 1)
            S = S_HU + S_H + S_U    # (N, T, J, 1)
            S = tf.squeeze(S, axis=[-1]) # (N, T, J)
            assert S_H.get_shape().as_list() == [None, T, 1, 1], "S_H: expected {}, got {}".format([None, T, 1, 1], S.get_shape().as_list())
            assert S_U.get_shape().as_list() == [None, 1, J, 1], "S_U: expected {}, got {}".format([None, 1, J, 1], S.get_shape().as_list())
            assert S_HU.get_shape().as_list() == [None, T, J, 1], "S_HU: expected {}, got {}".format([None, T, J, 1], S.get_shape().as_list())
            assert S.get_shape().as_list() == [None, T, J], "S: expected {}, got {}".format([None, T, J], S.get_shape().as_list())

        # Context-to-query (C2Q) Attention
        with vs.variable_scope("C2Q"):
            _, a_c2q = masked_softmax(S, HU_mask_new, dim=-1)  # Softmax for each context t, (N, T, J)
            assert a_c2q.get_shape().as_list() == [None, T, J], "a_c2q: expected {}, got {}".format([None, T, J], a_c2q.get_shape().as_list())

            a_c2q = tf.expand_dims(a_c2q, axis=-1)  # (N, T, J) -> (N, T, J, 1)
            assert a_c2q.get_shape().as_list() == [None, T, J, 1], "a_c2q: expected {}, got {}".format([None, T, J, 1], a_c2q.get_shape().as_list())

            attn_c2q = tf.reduce_sum(a_c2q * U_new, axis=2) # reduce across the J-dimension: (N, T, J, d) -> (N, T, d)
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

    def build_graph_old(self, H, H_mask, U, U_mask, d):
        """
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

        # Reshape H and U to be both (N, T, J, d)
        H_new = tf.tile(tf.reshape(H, [-1, T, 1, d]), [1, 1, J, 1]) # H: (N, T, d) -> (N, T, 1, d) -> (N, T, J, d)
        U_new = tf.tile(tf.reshape(U, [-1, 1, J, d]), [1, T, 1, 1]) # U: (N, J, d) -> (N, 1, J, d) -> (N, T, J, d)
        # Reshape H_mask and U_mask to be both (N, T, J)
        H_mask_new = tf.tile(tf.reshape(H_mask, [-1, T, 1]), [1, 1, J]) # H_mask: (N, T) -> (N, T, 1) -> (N, T, J)
        U_mask_new = tf.tile(tf.reshape(U_mask, [-1, 1, J]), [1, T, 1]) # U_mask: (N, J) -> (N, 1, J) -> (N, T, J)
        HU_mask_new = tf.cast(tf.cast(H_mask_new, tf.bool) & tf.cast(U_mask_new, tf.bool), tf.int32) # (N, T, J)

        assert H_new.get_shape().as_list() == [None, T, J, d], "H_new: expected {}, got {}".format([None, T, J, d], H_new.get_shape().as_list())
        assert U_new.get_shape().as_list() == [None, T, J, d], "U_new: expected {}, got {}".format([None, T, J, d], U_new.get_shape().as_list())
        assert H_mask_new.get_shape().as_list() == [None, T, J], "H_mask_new: expected {}, got {}".format([None, T, J], H_mask_new.get_shape().as_list())
        assert U_mask_new.get_shape().as_list() == [None, T, J], "U_mask_new: expected {}, got {}".format([None, T, J], U_mask_new.get_shape().as_list())
        assert HU_mask_new.get_shape().as_list() == [None, T, J], "HU_mask_new: expected {}, got {}".format([None, T, J], HU_mask_new.get_shape().as_list())

        # Contruct the similarity matrix S, dimension: (N, T, J)
        with vs.variable_scope("AttnSimilarity"):
            # S = tf.contrib.layers.fully_connected(HU_concat, num_outputs=1, activation_fn=None) # (N, T, J, 1)
            # S = tf.squeeze(S, axis=[3]) # (N, T, J)
            S = get_logits(args=[H_new, U_new, H_new * U_new], output_size=1, bias=None, input_keep_prob=self.keep_prob, is_train=(self.keep_prob<1.0))
            assert S.get_shape().as_list() == [None, T, J], "S: expected {}, got {}".format([None, T, J], S.get_shape().as_list())

        # Context-to-query (C2Q) Attention
        with vs.variable_scope("C2Q"):
            _, a_c2q = masked_softmax(S, HU_mask_new, dim=-1)  # Softmax for each context t, (N, T, J)
            assert a_c2q.get_shape().as_list() == [None, T, J], "a_c2q: expected {}, got {}".format([None, T, J], a_c2q.get_shape().as_list())

            a_c2q = tf.tile(tf.reshape(a_c2q, [-1, T, J, 1]), [1, 1, 1, d])  # (N, T, J, d)
            assert a_c2q.get_shape().as_list() == [None, T, J, d], "a_c2q: expected {}, got {}".format([None, T, J, d], a_c2q.get_shape().as_list())

            attn_c2q = tf.reduce_sum(tf.multiply(a_c2q, U_new), axis=2) # reduce across the J-dimension -> (N, T, d)
            assert attn_c2q.get_shape().as_list() == [None, T, d], "attn_c2q: expected {}, got {}".format([None, T, d], attn_c2q.get_shape().as_list())

        # Query-to-context (Q2C) Attention
        with vs.variable_scope("Q2C"):
            b = tf.reduce_max(S, axis=-1) # reduce across the J-dimension -> (N, T)
            assert b.get_shape().as_list() == [None, T], "b: expected {}, got {}".format([None, T], b.get_shape().as_list())

            _, b = masked_softmax(b, H_mask, dim=-1) # (N, T)
            assert b.get_shape().as_list() == [None, T], "b: expected {}, got {}".format([None, T], b.get_shape().as_list())

            b = tf.tile(tf.reshape(b, [-1, T, 1]), [1, 1, d]) # (N, T, d)
            assert b.get_shape().as_list() == [None, T, d], "b: expected {}, got {}".format([None, T, d], b.get_shape().as_list())

            attn_q2c = tf.reduce_sum(tf.multiply(b, H), axis=1) # reduce across the T-dimension -> (N, d)
            attn_q2c = tf.tile(tf.reshape(attn_q2c, [-1, 1, d]), [1, T, 1]) # (N, d) -> (N, 1, d) -> (N, T, d)
            assert attn_q2c.get_shape().as_list() == [None, T, d], "attn_q2c: expected {}, got {}".format([None, T, d], attn_q2c.get_shape().as_list())

        # Generate contextual embeddings using both attn_c2q and attn_q2c
        with vs.variable_scope("AttentionConcat"):
            attn_result = tf.concat([H, attn_c2q, tf.multiply(H, attn_c2q), tf.multiply(H, attn_q2c)], -1) # (N, T, 4 * d)
            assert attn_result.get_shape().as_list() == [None, T, 4*d], "attn_result: expected {}, got {}".format([None, T, 4*d], attn_result.get_shape().as_list())

        return attn_result


class BidafModeling(object):
    """
        Module for LSTM Modeling Layer
    """

    def __init__(self, hidden_size, keep_prob):
        self.keep_prob = keep_prob
        self.hidden_size = hidden_size

    def modeling_lstm(self, inputs, input_lens):
        rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        rnn_cell_fw = DropoutWrapper(rnn_cell_fw, input_keep_prob=self.keep_prob)
        rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        rnn_cell_bw = DropoutWrapper(rnn_cell_bw, input_keep_prob=self.keep_prob)

        (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

        out = tf.concat([fw_out, bw_out], 2)
        out = tf.nn.dropout(out, self.keep_prob)
        return out

    def build_graph(self, G, G_mask):
        """
        Inputs:
            G: (None, context_len, 8 * hidden_size)
            G_mask: (None, context_len)
        Returns:
            M: (None, context_len, 2 * hidden_size)
        """
        input_lens = tf.reduce_sum(G_mask, reduction_indices=1)

        with tf.variable_scope('FirstLayer'):
            temp = self.modeling_lstm(G, input_lens)
        with tf.variable_scope('SecondLayer'):
            M = self.modeling_lstm(temp, input_lens)

        return M


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
