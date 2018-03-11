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

"""This file defines the top-level model"""



import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops, rnn_cell

from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator
from pretty_print import print_example
from modules import RNNEncoder, SimpleSoftmaxLayer, BasicAttn, masked_softmax
from modules import BidafAttention, SelfAttention, LSTM_Mapper
from util import VariationalDropout, TriLinearSim
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

logging.basicConfig(level=logging.INFO)


class QAModel(object):
    """Top-level Question Answering module"""

    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        """
        Initializes the QA model.

        Inputs:
          FLAGS: the flags passed in from main.py
          id2word: dictionary mapping word idx (int) to word (string)
          word2id: dictionary mapping word (string) to word idx (int)
          emb_matrix: numpy array shape (400002, embedding_size) containing pre-traing GloVe embeddings
        """
        print ("Initializing the QAModel...")
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id
        self.batcher = Batcher(os.path.join(self.FLAGS.data_dir, 'elmo_voca.txt'), self.FLAGS.max_word_size)
        # parse the map that map pos tag to pos id
        self.pos_tag_id_map = {}
        with open(os.path.join(self.FLAGS.main_dir, 'pos_tags.txt')) as f:
            pos_tag_lines = f.readlines()
        for i in range(len(pos_tag_lines)):
            self.pos_tag_id_map[pos_tag_lines[i][:-1]] = i + 1 # need to get rid of the trailing newline character
        # get the NE tag to id
        self.ne_tag_id_map = {}
        all_NE_tag = ['B-FACILITY', 'B-GPE', 'B-GSP', 'B-LOCATION', 'B-ORGANIZATION', 'B-PERSON', 'I-FACILITY', 'I-GPE', 'I-GSP', 'I-LOCATION', 'I-ORGANIZATION', 'I-PERSON','O'] # I know this not elegant
        for i in range(len(all_NE_tag)):
            self.ne_tag_id_map[all_NE_tag[i]] = i + 1

        # filters for the char cnn
        self.filters = [(1, 4), (2, 8), (3, 16), (4, 32), (5, 64)]

        # Add all parts of the graph
        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix)
            self.build_graph()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(list(zip(clipped_gradients, params)), global_step=self.global_step)

        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()


    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs.
        # These are all batch-first: the None corresponds to batch_size and
        # allows you to run the same model with variable batch_size
        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        # NOTE: CHANGE
        self.context_char = tf.placeholder(tf.int32, shape=[None, None, self.FLAGS.max_word_size])
        self.context_pos_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.context_ne_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])

        self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        # NOTE: CHANGE
        self.qn_char = tf.placeholder(tf.int32, shape=[None, None, self.FLAGS.max_word_size])
        self.qn_pos_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.qn_ne_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])

        self.context_EM = tf.placeholder(tf.float32, shape=[None, self.FLAGS.context_len, 1])
        self.qn_EM = tf.placeholder(tf.float32, shape=[None, self.FLAGS.question_len, 1])

        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])

        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())


    def add_char_embedding_layer(self):
        with vs.variable_scope("embed_char"):
            char_embedding_matrix = tf.get_variable(name='char_emb_matrix', shape=[self.FLAGS.num_of_char, self.FLAGS.char_embedding_size], initializer=tf.initializers.random_uniform(minval=-0.5, maxval=0.5, dtype=tf.float32))
            context_char_embedding = embedding_ops.embedding_lookup(char_embedding_matrix, self.context_char)
            qn_char_embedding = embedding_ops.embedding_lookup(char_embedding_matrix, self.qn_char)

            def make_convolutions(inp, filters):
                convolutions = []
                for i, (width, num) in enumerate(filters):
                    w_init = tf.random_uniform_initializer(
                            minval=-0.05, maxval=0.05)
                    w = tf.get_variable(
                        "W_cnn_%s" % i,
                        [1, width, self.FLAGS.char_embedding_size, num],
                        initializer=w_init,
                        dtype=tf.float32)
                    b = tf.get_variable(
                        "b_cnn_%s" % i, [num], dtype=tf.float32,
                        initializer=tf.constant_initializer(0.0))

                    conv = tf.nn.conv2d(inp, w, strides=[1, 1, 1, 1], padding="VALID") + b
                    # now max pool
                    conv = tf.nn.max_pool(conv, [1, 1, self.FLAGS.max_word_size - width + 1, 1],[1, 1, 1, 1], 'VALID')

                    # activation
                    conv = tf.nn.relu(conv)
                    conv = tf.squeeze(conv, squeeze_dims=[2])
                    convolutions.append(conv)
                return tf.concat(convolutions, 2)

            self.context_embs_char = make_convolutions(context_char_embedding, self.filters)
            tf.get_variable_scope().reuse_variables()
            self.qn_embs_char = make_convolutions(qn_char_embedding, self.filters)

    def add_embedding_layer(self, emb_matrix):
        """
        Adds word embedding layer to the graph.

        Inputs:
          emb_matrix: shape (400002, embedding_size).
            The GloVe vectors, plus vectors for PAD and UNK.
        """
        with vs.variable_scope("embed_glove"):

            # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)

            # Get the word embeddings for the context and question,
            # using the placeholders self.context_ids and self.qn_ids
            self.context_embs_glove = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids) # shape (batch_size, context_len, embedding_size)
            self.qn_embs_glove = embedding_ops.embedding_lookup(embedding_matrix, self.qn_ids) # shape (batch_size, question_len, embedding_size)

        with vs.variable_scope("embed_pos"):
            pos_embedding_matrix = tf.get_variable(name='pos_emb_matrix', shape=[len(self.pos_tag_id_map) + 1, self.FLAGS.pos_embedding_size], initializer=tf.initializers.random_uniform(minval=-0.5, maxval=0.5, dtype=tf.float32))
            self.context_embs_pos = embedding_ops.embedding_lookup(pos_embedding_matrix, self.context_pos_ids) # shape (batch_size, context_len, pos_embedding_size)
            self.qn_embs_pos = embedding_ops.embedding_lookup(pos_embedding_matrix, self.qn_pos_ids) # shape (batch_size, question_len, pos_embedding_size)

        with vs.variable_scope("ne_pos"):
            ne_embedding_matrix = tf.get_variable(name='ne_emb_matrix', shape=[len(self.ne_tag_id_map) + 1, self.FLAGS.ne_embedding_size], initializer=tf.initializers.random_uniform(minval=-0.5, maxval=0.5, dtype=tf.float32))
            self.context_embs_ne = embedding_ops.embedding_lookup(pos_embedding_matrix, self.context_ne_ids) # shape (batch_size, context_len, ne_embedding_size)
            self.qn_embs_ne = embedding_ops.embedding_lookup(pos_embedding_matrix, self.qn_ne_ids) # shape (batch_size, question_len, ne_embedding_size)

        self.add_char_embedding_layer()

    # NOTE: CHANGE
    def add_embedding_layer_elmo(self, options_file, weight_file):
        """
        Adds word embedding layer to the graph.

        Inputs:
          options_file and weight_file for the pretrained elmo model
        """
        # Build the biLM graph.
        bilm = BidirectionalLanguageModel(options_file, weight_file)

        # Get ops to compute the LM embeddings.
        context_embeddings_op = bilm(self.context_elmo)
        question_embeddings_op = bilm(self.qn_elmo)

        # Get an op to compute ELMo (weighted average of the internal biLM layers)
        # Our SQuAD model includes ELMo at both the input and output layers
        # of the task GRU, so we need 4x ELMo representations for the question
        # and context at each of the input and output.
        # We use the same ELMo weights for both the question and context
        # at each of the input and output.
        self.context_embs_elmo = weight_layers('input', context_embeddings_op, l2_coef=0.0)['weighted_op'] # shape(batch size, context size, 32)
        with tf.variable_scope('', reuse=True):
            # the reuse=True scope reuses weights from the context for the question
            self.qn_embs_elmo = weight_layers(
                'input', question_embeddings_op, l2_coef=0.0
            )['weighted_op']
            # shape(batch size, question len, 32)

    def build_graph(self):
        # Similar to an implementation by AllenAI
        def VariationalEncoder(context, question, context_mask, question_mask, hidden_size, keep_prob):
            context = VariationalDropout(context, keep_prob)
            question = VariationalDropout(question, keep_prob)

            context_lens = tf.reduce_sum(context_mask, reduction_indices=1) # shape (batch_size)
            question_lens = tf.reduce_sum(question_mask, reduction_indices=1)

            rnn_cell_fw = rnn_cell.LSTMCell(hidden_size)
            rnn_cell_bw = rnn_cell.LSTMCell(hidden_size)
            (context_fw_out, context_bw_out), _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, context, context_lens, dtype=tf.float32)
            (question_fw_out, question_bw_out), _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, question, question_lens, dtype=tf.float32)

            context_hiddens = tf.concat([context_fw_out, context_bw_out], 2)
            question_hiddens = tf.concat([question_fw_out, question_bw_out], 2)

            context_hiddens = VariationalDropout(context_hiddens, keep_prob)
            question_hiddens = VariationalDropout(question_hiddens, keep_prob)
            return context_hiddens, question_hiddens

        T = self.FLAGS.context_len
        d = self.FLAGS.hidden_size

        # NOTE CHANGE: concantanate glove and elmo embedding
        # TODO: is the mask correct?
        self.context_embs = tf.concat([self.context_embs_glove, self.context_embs_pos, self.context_embs_ne, self.context_embs_char, self.context_EM], 2)
        self.qn_embs = tf.concat([self.qn_embs_glove, self.qn_embs_pos, self.qn_embs_ne, self.qn_embs_char, self.qn_EM], 2)

        with tf.variable_scope("Encoder"):  # Contextual Embedding Layer: Use an RNN Encoder to get hidden states for the context and the question
            context_hiddens, question_hiddens = VariationalEncoder(self.context_embs, self.qn_embs, self.context_mask, self.qn_mask, d, self.keep_prob)

        with tf.variable_scope("BidafAttention"):   # Bi-directional Attention Flow: (context_hiddens, question_hiddens) -> G (query-aware context representations)
            bidaf_attn_layer = BidafAttention(self.keep_prob)
            G = bidaf_attn_layer.build_graph(context_hiddens, self.context_mask, question_hiddens, self.qn_mask, 2*d) # (N, T, 8*d)
            assert G.get_shape().as_list() == [None, T, 8*d]

        with tf.variable_scope("MatchEncoder"):
            M = tf.contrib.layers.fully_connected(G, num_outputs=2*d, activation_fn=tf.nn.relu) # (N, T, 2*d)
            with tf.variable_scope("SelfAttention"):    # Generate SelfAttn
                # Apply dropout
                Q = VariationalDropout(M, self.keep_prob)

                # Go through recurrent layer and apply dropout
                SelfAttnMapper = LSTM_Mapper(d, False, self.keep_prob)
                Q = SelfAttnMapper.build_graph(Q, self.context_mask)
                Q = VariationalDropout(Q, self.keep_prob)   # (N, T, 2*d)
                assert Q.get_shape().as_list() == [None, T, 2*d]

                # (N, T, T) mask
                mask_bool = tf.cast(self.context_mask, tf.bool)
                mask2d = tf.expand_dims(mask_bool, axis=1) & tf.expand_dims(mask_bool, axis=2)    #  (N, T, T)
                assert mask2d.get_shape().as_list() == [None, T, T]

                # Obtain the raw similarity matrix
                sim = TriLinearSim(Q, Q)    # (N, T, T)
                assert sim.get_shape().as_list() == [None, T, T]

                # Force a word not to align with itself
                sim_mask_1 = tf.expand_dims(tf.eye(num_rows=T, dtype=tf.float32), axis=0) * (-1e29) # (1, T, T)
                sim_mask_2 = (1 - tf.cast(mask2d, 'float')) * (-1e30)  # (N, T, T)
                sim = sim + sim_mask_1 + sim_mask_2 # (N, T, T)
                assert sim.get_shape().as_list() == [None, T, T]

                # Allow zero-attention by adding a learned bias to the normalizer. Apply softmax.
                bias = tf.exp(tf.get_variable("no-alignment-bias", initializer=tf.constant(-1.0, dtype=tf.float32)))
                sim = tf.exp(sim)
                sim = sim / (tf.reduce_sum(sim, axis=2, keep_dims=True) + bias)
                assert sim.get_shape().as_list() == [None, T, T]

                # Obtain the final SelfAttn vector
                SelfAttn = tf.matmul(sim, Q)    # (N, T, 2*d)
                assert SelfAttn.get_shape().as_list() == [None, T, 2*d]

                # Concatenate
                SelfAttn = tf.concat([SelfAttn, Q, SelfAttn * Q], axis=-1)  # (N, T, 6*d)
                assert SelfAttn.get_shape().as_list() == [None, T, 6*d]

                # Fully connected layer
                SelfAttn = tf.contrib.layers.fully_connected(SelfAttn, num_outputs=2*d, activation_fn=tf.nn.relu) # (N, T, 2*d)
                assert SelfAttn.get_shape().as_list() == [None, T, 2*d]

            M += SelfAttn
            M = VariationalDropout(M, self.keep_prob)

        with tf.variable_scope("OutputMapper"):
            StartMapper = LSTM_Mapper(d, False, self.keep_prob)
            M_start = StartMapper.build_graph(M, self.context_mask)
            assert M_start.get_shape().as_list() == [None, T, 2*d]

        with vs.variable_scope("StartDist"):    # Use softmax layer to compute probability distribution for start location -> probdist_start: (batch_size, context_len)
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(M_start, self.context_mask)

        with vs.variable_scope("EndDist"):  # Use softmax layer to compute probability distribution for end location -> probdist_end: (batch_size, context_len)
            M_end = tf.concat([M, M_start], axis=-1)
            EndMapper = LSTM_Mapper(d, False, self.keep_prob)
            M_end = EndMapper.build_graph(M_end, self.context_mask)
            assert M_end.get_shape().as_list() == [None, T, 2*d]

            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(M_end, self.context_mask)


    def build_graph_v2(self):
        T = self.FLAGS.context_len
        d = self.FLAGS.hidden_size

        # NOTE CHANGE: concantanate glove and elmo embedding
        # TODO: is the mask correct?
        self.context_embs = tf.concat([self.context_embs_glove, self.context_embs_pos, self.context_embs_ne, self.context_embs_char, self.context_EM], 2)
        self.qn_embs = tf.concat([self.qn_embs_glove, self.qn_embs_pos, self.qn_embs_ne, self.qn_embs_char, self.qn_EM], 2)

        with tf.variable_scope("Encoder"):  # Contextual Embedding Layer: Use an RNN Encoder to get hidden states for the context and the question
            encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            context_hiddens = encoder.build_graph(self.context_embs, self.context_mask, scope='context') # (batch_size, context_len, hidden_size*2)
            if self.FLAGS.share_LSTM_weights:
                tf.get_variable_scope().reuse_variables()
                question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask, scope='context') # (batch_size, qn_len, hidden_size*2)
            else:
                question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask, scope='question') # (batch_size, qn_len, hidden_size*2)

        with tf.variable_scope("BidafAttention"):   # Bi-directional Attention Flow: (context_hiddens, question_hiddens) -> G (query-aware context representations)
            bidaf_attn_layer = BidafAttention(self.keep_prob)
            G = bidaf_attn_layer.build_graph(context_hiddens, self.context_mask, question_hiddens, self.qn_mask, 2 * self.FLAGS.hidden_size)

        with tf.variable_scope("BidafModeling"):    # Blend the Bidaf representation using a Bi-LSTM (G -> M1)
            BidafMapper = LSTM_Mapper(self.FLAGS.hidden_size, True, self.keep_prob)
            M1 = BidafMapper.build_graph(G, self.context_mask) # (N, T, 2*d)

        with tf.variable_scope("SelfAttention"):    # Apply self-attention to M1 -> self_attn_vecs
            self_attn_layer = SelfAttention(self.FLAGS.hidden_size, self.keep_prob)
            self_attn_vecs = self_attn_layer.build_graph(M1, self.context_mask) # (N, T, 6*d)
            self_attn_vecs = VariationalDropout(self_attn_vecs, self.keep_prob) # (N, T, 6*d)

        selfattn_highway = False
        with tf.variable_scope("SelfAttnModeling"): # Blend the self-attended representation (self_attn_vecs -> M2)
            if selfattn_highway:
                self_attn_vecs = tf.contrib.layers.fully_connected(self_attn_vecs, num_outputs=2*d, activation_fn=tf.nn.relu) # (N, T, 2*d)
                self_attn_vecs = VariationalDropout(self_attn_vecs, self.keep_prob)

                SelfAttnMapper = LSTM_Mapper(self.FLAGS.hidden_size, False, self.keep_prob)
                M2 = SelfAttnMapper.build_graph(self_attn_vecs, self.context_mask)
                M2 = VariationalDropout(M2, self.keep_prob)

                transform_gate = tf.contrib.layers.fully_connected(self_attn_vecs, num_outputs=2*d, activation_fn=tf.nn.sigmoid) # (N, T, 2*d)
                M2 = transform_gate * M2 + (1 - transform_gate) * self_attn_vecs # (N, T, 2*d)
            else:
                SelfAttnMapper = LSTM_Mapper(self.FLAGS.hidden_size, True, self.keep_prob)
                M2 = SelfAttnMapper.build_graph(self_attn_vecs, self.context_mask)
                M2 = tf.nn.dropout(M2, self.keep_prob)

            assert M2.get_shape().as_list() == [None, T, 2*d]

        with vs.variable_scope("StartDist"):    # Use softmax layer to compute probability distribution for start location -> probdist_start: (batch_size, context_len)
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(M2, self.context_mask)

        with vs.variable_scope("EndDist"):  # Use softmax layer to compute probability distribution for end location -> probdist_end: (batch_size, context_len)
            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(M2, self.context_mask)


    def add_loss(self):
        """
        Add loss computation to the graph.

        Uses:
          self.logits_start: shape (batch_size, context_len)
            IMPORTANT: Assumes that self.logits_start is masked (i.e. has -large in masked locations).
            That's because the tf.nn.sparse_softmax_cross_entropy_with_logits
            function applies softmax and then computes cross-entropy loss.
            So you need to apply masking to the logits (by subtracting large
            number in the padding location) BEFORE you pass to the
            sparse_softmax_cross_entropy_with_logits function.

          self.ans_span: shape (batch_size, 2)
            Contains the gold start and end locations

        Defines:
          self.loss_start, self.loss_end, self.loss: all scalar tensors
        """
        with vs.variable_scope("loss"):

            # Calculate loss for prediction of start position
            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start, labels=self.ans_span[:, 0]) # loss_start has shape (batch_size)
            self.loss_start = tf.reduce_mean(loss_start) # scalar. avg across batch
            tf.summary.scalar('loss_start', self.loss_start) # log to tensorboard

            # Calculate loss for prediction of end position
            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end, labels=self.ans_span[:, 1])
            self.loss_end = tf.reduce_mean(loss_end)
            tf.summary.scalar('loss_end', self.loss_end)

            # Add the two losses
            self.loss = self.loss_start + self.loss_end
            tf.summary.scalar('loss', self.loss)


    def run_train_iter(self, session, batch, summary_writer):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout # apply dropout
        input_feed[self.context_char] = batch.context_char
        input_feed[self.qn_char] = batch.qn_char
        input_feed[self.context_pos_ids] = batch.context_pos_ids
        input_feed[self.qn_pos_ids] = batch.qn_pos_ids
        input_feed[self.context_ne_ids] = batch.context_ne_ids
        input_feed[self.qn_ne_ids] = batch.qn_ne_ids
        input_feed[self.context_EM] = np.expand_dims(batch.context_em, axis=-1)
        input_feed[self.qn_EM] = np.zeros((batch.context_em.shape[0], self.FLAGS.question_len, 1)) # padding

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


    def get_loss(self, session, batch):
        """
        Run forward-pass only; get loss.

        Inputs:
          session: TensorFlow session
          batch: a Batch object

        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.context_char] = batch.context_char
        input_feed[self.qn_char] = batch.qn_char
        input_feed[self.context_pos_ids] = batch.context_pos_ids
        input_feed[self.qn_pos_ids] = batch.qn_pos_ids
        input_feed[self.context_ne_ids] = batch.context_ne_ids
        input_feed[self.qn_ne_ids] = batch.qn_ne_ids
        input_feed[self.context_EM] = np.expand_dims(batch.context_em, axis=-1)
        input_feed[self.qn_EM] = np.zeros((batch.context_em.shape[0], self.FLAGS.question_len, 1)) # padding
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss


    def get_prob_dists(self, session, batch):
        """
        Run forward-pass only; get probability distributions for start and end positions.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          probdist_start and probdist_end: both shape (batch_size, context_len)
        """
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.context_char] = batch.context_char
        input_feed[self.qn_char] = batch.qn_char
        input_feed[self.context_pos_ids] = batch.context_pos_ids
        input_feed[self.qn_pos_ids] = batch.qn_pos_ids
        input_feed[self.context_ne_ids] = batch.context_ne_ids
        input_feed[self.qn_ne_ids] = batch.qn_ne_ids
        input_feed[self.context_EM] = np.expand_dims(batch.context_em, axis=-1)
        input_feed[self.qn_EM] = np.zeros((batch.context_em.shape[0], self.FLAGS.question_len, 1)) # padding
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.probdist_start, self.probdist_end]
        [probdist_start, probdist_end] = session.run(output_feed, input_feed)
        return probdist_start, probdist_end


    def get_start_end_pos(self, session, batch):
            """
            Run forward-pass only; get the most likely answer span.

            Inputs:
              session: TensorFlow session
              batch: Batch object

            Returns:
              start_pos, end_pos: both numpy arrays shape (batch_size).
                The most likely start and end positions for each example in the batch.
            """
            # Get start_dist and end_dist, both shape (batch_size, context_len)
            start_dist, end_dist = self.get_prob_dists(session, batch)
            batch_size, context_len = start_dist.shape
            start_dist = np.expand_dims(start_dist,2)
            end_dist = np.expand_dims(end_dist,1)
            dist_mat =np.matmul(start_dist,end_dist)

            mask = np.triu(np.ones((context_len,context_len)))
            mask = np.tril(mask,13)
            mask = np.repeat(mask[:,:,np.newaxis],batch_size,axis=2)
            mask = np.transpose(mask,(2,0,1))
            assert dist_mat.shape==mask.shape and mask.shape==(batch_size,context_len,context_len), "expected {}, got dist {}, mask {}".format((batch_size,context_len,context_len), dist_mat.shape,mask.shape)
            prob_mat = dist_mat*mask

            start_pos = np.argmax(np.amax(prob_mat,axis=2),axis=1)
            end_pos = np.argmax(np.amax(prob_mat,axis=1),axis=1)

            # start_pos = np.argmax(start_dist, axis=1)
            # end_pos = np.argmax(end_dist, axis=1)

            return start_pos, end_pos


    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path):
        """
        Get loss for entire dev set.

        Inputs:
          session: TensorFlow session
          dev_qn_path, dev_context_path, dev_ans_path: paths to the dev.{context/question/answer} data files

        Outputs:
          dev_loss: float. Average loss across the dev set.
        """
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []
        context_pos_path = dev_context_path + '.pos'
        qn_pos_path = dev_qn_path + '.pos'
        context_ne_path = dev_context_path + '.ne'
        qn_ne_path = dev_qn_path + '.ne'

        # Iterate over dev set batches
        # Note: here we set discard_long=True, meaning we discard any examples
        # which are longer than our context_len or question_len.
        # We need to do this because if, for example, the true answer is cut
        # off the context, then the loss function is undefined.
        for batch in get_batch_generator(self.word2id, dev_context_path, dev_qn_path, dev_ans_path, context_pos_path, qn_pos_path, context_ne_path, qn_ne_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True, batcher=self.batcher):

            # Get loss for this batch
            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print ("Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic))

        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss


    def check_f1_em(self, session, context_path, qn_path, ans_path, dataset, num_samples=100, print_to_screen=False):
        """
        Sample from the provided (train/dev) set.
        For each sample, calculate F1 and EM score.
        Return average F1 and EM score for all samples.
        Optionally pretty-print examples.

        Note: This function is not quite the same as the F1/EM numbers you get from "official_eval" mode.
        This function uses the pre-processed version of the e.g. dev set for speed,
        whereas "official_eval" mode uses the original JSON. Therefore:
          1. official_eval takes your max F1/EM score w.r.t. the three reference answers,
            whereas this function compares to just the first answer (which is what's saved in the preprocessed data)
          2. Our preprocessed version of the dev set is missing some examples
            due to tokenization issues (see squad_preprocess.py).
            "official_eval" includes all examples.

        Inputs:
          session: TensorFlow session
          qn_path, context_path, ans_path: paths to {dev/train}.{question/context/answer} data files.
          dataset: string. Either "train" or "dev". Just for logging purposes.
          num_samples: int. How many samples to use. If num_samples=0 then do whole dataset.
          print_to_screen: if True, pretty-prints each example to screen

        Returns:
          F1 and EM: Scalars. The average across the sampled examples.
        """
        logging.info("Calculating F1/EM for %s examples in %s set..." % (str(num_samples) if num_samples != 0 else "all", dataset))

        f1_total = 0.
        em_total = 0.
        example_num = 0

        context_pos_path = context_path + '.pos'
        qn_pos_path = qn_path + '.pos'
        context_ne_path = context_path + '.ne'
        qn_ne_path = qn_path + '.ne'

        tic = time.time()

        # Note here we select discard_long=False because we want to sample from the entire dataset
        # That means we're truncating, rather than discarding, examples with too-long context or questions
        for batch in get_batch_generator(self.word2id, context_path, qn_path, ans_path, context_pos_path, qn_pos_path, context_ne_path, qn_ne_path,self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=False, batcher=self.batcher):

            pred_start_pos, pred_end_pos = self.get_start_end_pos(session, batch)

            # Convert the start and end positions to lists length batch_size
            pred_start_pos = pred_start_pos.tolist() # list length batch_size
            pred_end_pos = pred_end_pos.tolist() # list length batch_size

            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                example_num += 1

                # Get the predicted answer
                # Important: batch.context_tokens contains the original words (no UNKs)
                # You need to use the original no-UNK version when measuring F1/EM
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)

                # Get true answer (no UNKs)
                true_answer = " ".join(true_ans_tokens)

                # Calc F1/EM
                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em

                # Optionally pretty-print
                if print_to_screen:
                    print_example(self.word2id, batch.context_tokens[ex_idx], batch.qn_tokens[ex_idx], batch.ans_span[ex_idx, 0], batch.ans_span[ex_idx, 1], pred_ans_start, pred_ans_end, true_answer, pred_answer, f1, em)

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= example_num
        em_total /= example_num

        toc = time.time()
        logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

        return f1_total, em_total


    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path):
        """
        Main training loop.

        Inputs:
          session: TensorFlow session
          {train/dev}_{qn/context/ans}_path: paths to {train/dev}.{context/question/answer} data files
        """

        # Print number of model parameters
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum([np.prod(tf.shape(t.value()).eval()) for t in params])
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # Log the hyperparameters so that we know exactly what we ran
        logging.info("---Hyperparameters---")
        logging.info("learning_rate: {}".format(self.FLAGS.learning_rate))
        logging.info("batch_size: {}".format(self.FLAGS.batch_size))
        logging.info("dropout: {}".format(self.FLAGS.dropout))
        logging.info("hidden_size: {}".format(self.FLAGS.hidden_size))
        logging.info("embedding_size: {}".format(self.FLAGS.embedding_size))
        logging.info("context_len: {}".format(self.FLAGS.context_len))
        logging.info("question_len: {}".format(self.FLAGS.question_len))
        logging.info("share_LSTM_weights: {}".format(self.FLAGS.share_LSTM_weights))
        logging.info("max_gradient_norm: {}".format(self.FLAGS.max_gradient_norm))
        logging.info("---------------------")

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        best_dev_f1 = None
        best_dev_em = None

        context_pos_path = train_context_path + '.pos'
        qn_pos_path = train_qn_path + '.pos'
        context_ne_path = train_context_path + '.ne'
        qn_ne_path = train_qn_path + '.ne'

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for batch in get_batch_generator(self.word2id, train_context_path, train_qn_path, train_ans_path, context_pos_path, qn_pos_path, context_ne_path, qn_ne_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True, batcher=self.batcher):

                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Update exponentially-smoothed loss
                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

                # Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.FLAGS.eval_every == 0:

                    # Get loss for entire dev set and log to tensorboard
                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                    # Get F1/EM on train set and log to tensorboard
                    train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)


                    # Get F1/EM on dev set and log to tensorboard
                    dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)


                    # Early stopping based on dev EM. You could switch this to use F1 instead.
                    # Based on observations, EM and F1 show almost 100% the same trend during training
                    if best_dev_f1 is None or dev_f1 > best_dev_f1:
                        best_dev_f1 = dev_f1
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()



def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)
