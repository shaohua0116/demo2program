from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.seq2seq as seq2seq
from collections import namedtuple
from tensorflow.python.layers import core as layer_core
from models.util import log
from models.ops import fc, conv2d


SequenceLossOutput = namedtuple(
    'SequenceLossOutput',
    'mask loss output token_acc seq_acc syntax_acc ' +
    'is_correct_syntax pred_tokens is_same_seq')


class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True, global_step=None):
        self.debug = debug_information
        self.global_step = global_step

        self.config = config
        self.dataset_type = config.dataset_type
        self.scheduled_sampling = \
            getattr(config, 'scheduled_sampling', False) or False
        self.scheduled_sampling_decay_steps = \
            getattr(config, 'scheduled_sampling_decay_steps', 5000) or 5000
        self.batch_size = config.batch_size
        self.encoder_rnn_type = config.encoder_rnn_type
        self.dataset_type = config.dataset_type
        self.dsl_type = config.dsl_type
        self.env_type = config.env_type
        self.vizdoom_pos_keys = config.vizdoom_pos_keys
        self.vizdoom_max_init_pos_len = config.vizdoom_max_init_pos_len
        self.perception_type = config.perception_type
        self.level = config.level
        self.num_lstm_cell_units = config.num_lstm_cell_units
        self.demo_aggregation = config.demo_aggregation
        self.dim_program_token = config.dim_program_token
        self.max_program_len = config.max_program_len
        self.max_demo_len = config.max_demo_len
        self.max_action_len = self.max_demo_len
        self.k = config.k
        self.test_k = config.test_k
        self.h = config.h
        self.w = config.w
        self.depth = config.depth
        self.action_space = config.action_space
        self.per_dim = config.per_dim

        if self.scheduled_sampling:
            if global_step is None:
                raise ValueError('scheduled sampling requires global_step')
            # linearly decaying sampling probability
            final_teacher_forcing_prob = 0.1
            self.sample_prob = tf.train.polynomial_decay(
                1.0, global_step, self.scheduled_sampling_decay_steps,
                end_learning_rate=final_teacher_forcing_prob,
                power=1.0, name='scheduled_sampling')

        # Text
        if self.dataset_type == 'karel':
            from karel_env.dsl import get_KarelDSL
            self.vocab = get_KarelDSL(dsl_type=self.dsl_type, seed=123)
        else:
            from vizdoom_env.dsl.vocab import VizDoomDSLVocab
            self.vocab = VizDoomDSLVocab(perception_type=self.perception_type,
                                         level=self.level)

        # create placeholders for the input
        self.program_id = tf.placeholder(
            name='program_id', dtype=tf.string,
            shape=[self.batch_size],
        )

        self.program = tf.placeholder(
            name='program', dtype=tf.float32,
            shape=[self.batch_size, self.dim_program_token,
                   self.max_program_len],
        )

        self.program_tokens = tf.placeholder(
            name='program_tokens', dtype=tf.int32,
            shape=[self.batch_size, self.max_program_len])

        self.s_h = tf.placeholder(
            name='s_h', dtype=tf.float32,
            shape=[self.batch_size, self.k, self.max_demo_len, self.h, self.w,
                   self.depth],
        )

        self.test_s_h = tf.placeholder(
            name='test_s_h', dtype=tf.float32,
            shape=[self.batch_size, self.test_k, self.max_demo_len, self.h, self.w,
                   self.depth],
        )

        self.a_h = tf.placeholder(
            name='a_h', dtype=tf.float32,
            shape=[self.batch_size, self.k, self.max_action_len,
                   self.action_space],
        )

        self.a_h_tokens = tf.placeholder(
            name='a_h_tokens', dtype=tf.int32,
            shape=[self.batch_size, self.k, self.max_action_len],
        )

        self.per = tf.placeholder(
            name='per', dtype=tf.float32,
            shape=[self.batch_size, self.k, self.max_demo_len, self.per_dim],
        )

        self.test_per = tf.placeholder(
            name='test_per', dtype=tf.float32,
            shape=[self.batch_size, self.test_k, self.max_demo_len, self.per_dim],
        )

        if self.config.dataset_type == 'vizdoom':
            self.init_pos = tf.placeholder(
                name='init_pos', dtype=tf.int32,
                shape=[self.batch_size, self.k, len(self.vizdoom_pos_keys),
                       self.vizdoom_max_init_pos_len, 2],
            )

            self.init_pos_len = tf.placeholder(
                name='init_pos_len', dtype=tf.int32,
                shape=[self.batch_size, self.k, len(self.vizdoom_pos_keys)],
            )

            self.test_init_pos = tf.placeholder(
                name='test_init_pos', dtype=tf.int32,
                shape=[self.batch_size, self.test_k, len(self.vizdoom_pos_keys),
                       self.vizdoom_max_init_pos_len, 2],
            )

            self.test_init_pos_len = tf.placeholder(
                name='test_init_pos_len', dtype=tf.int32,
                shape=[self.batch_size, self.test_k, len(self.vizdoom_pos_keys)],
            )
        else:
            self.init_pos = None
            self.init_pos_len = None
            self.test_init_pos = None
            self.test_init_pos_len = None

        self.program_len = tf.placeholder(
            name='program_len', dtype=tf.float32,
            shape=[self.batch_size, 1],
        )
        self.program_len = tf.cast(self.program_len, dtype=tf.int32)

        self.demo_len = tf.placeholder(
            name='demo_len', dtype=tf.float32,
            shape=[self.batch_size, self.k],
        )
        self.demo_len = tf.cast(self.demo_len, dtype=tf.int32)

        self.test_demo_len = tf.placeholder(
            name='test_demo_len', dtype=tf.float32,
            shape=[self.batch_size, self.test_k],
        )
        self.test_demo_len = tf.cast(self.test_demo_len, dtype=tf.int32)

        self.action_len = self.demo_len

        self.is_train = tf.placeholder(
            name='is_train', dtype=tf.bool,
            shape=[],
        )

        self.is_training = tf.placeholder_with_default(
            bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=True):
        fd = {
            self.program_id: batch_chunk['id'],
            self.program: batch_chunk['program'],
            self.program_tokens: batch_chunk['program_tokens'],
            self.s_h: batch_chunk['s_h'],
            self.a_h: batch_chunk['a_h'],
            self.a_h_tokens: batch_chunk['a_h_tokens'],
            self.program_len: batch_chunk['program_len'],
            self.demo_len: batch_chunk['demo_len'],
            self.test_s_h: batch_chunk['test_s_h'],
            self.test_demo_len: batch_chunk['test_demo_len'],
            self.per: batch_chunk['per'],
            self.test_per: batch_chunk['test_per'],
            self.is_train: is_training
        }
        if self.dataset_type == 'vizdoom':
            fd[self.init_pos] = batch_chunk['init_pos']
            fd[self.init_pos_len] = batch_chunk['init_pos_len']
            fd[self.test_init_pos] = batch_chunk['test_init_pos']
            fd[self.test_init_pos_len] = batch_chunk['test_init_pos_len']
        return fd

    def build(self, is_train=True):
        max_demo_len = self.max_demo_len
        demo_len = self.demo_len
        s_h = self.s_h
        depth = self.depth

        # s [bs, h, w, depth] -> feature [bs, v]
        # CNN
        def State_Encoder(s, batch_size, scope='State_Encoder', reuse=False):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warning(scope.name)
                _ = conv2d(s, 16, is_train, k_h=3, k_w=3,
                           info=not reuse, batch_norm=True, name='conv1')
                _ = conv2d(_, 32, is_train, k_h=3, k_w=3,
                           info=not reuse, batch_norm=True, name='conv2')
                _ = conv2d(_, 48, is_train, k_h=3, k_w=3,
                           info=not reuse, batch_norm=True, name='conv3')
                if self.dataset_type == 'vizdoom':
                    _ = conv2d(_, 48, is_train, k_h=3, k_w=3,
                               info=not reuse, batch_norm=True, name='conv4')
                    _ = conv2d(_, 48, is_train, k_h=3, k_w=3,
                               info=not reuse, batch_norm=True, name='conv5')
                state_feature = tf.reshape(_, [batch_size, -1])
                return state_feature

        # s_h [bs, t, h, w, depth] -> feature [bs, v]
        # LSTM
        def Demo_Encoder(s_h, seq_lengths, scope='Demo_Encoder', reuse=False):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warning(scope.name)
                state_features = tf.reshape(
                    State_Encoder(tf.reshape(s_h, [-1, self.h, self.w, depth]),
                                  self.batch_size * max_demo_len, reuse=reuse),
                    [self.batch_size, max_demo_len, -1])

                if self.encoder_rnn_type == 'lstm':
                    cell = rnn.BasicLSTMCell(
                        num_units=self.num_lstm_cell_units,
                        state_is_tuple=True)
                elif self.encoder_rnn_type == 'rnn':
                    cell = rnn.BasicRNNCell(num_units=self.num_lstm_cell_units)
                elif self.encoder_rnn_type == 'gru':
                    cell = rnn.GRUCell(num_units=self.num_lstm_cell_units)
                else:
                    raise ValueError('Unknown encoder rnn type')

                new_h, cell_state = tf.nn.dynamic_rnn(
                    cell=cell, dtype=tf.float32, sequence_length=seq_lengths,
                    inputs=state_features)
                all_states = new_h
                return all_states, cell_state.h, cell_state.c

        def SecondPathEncoder(prev_h, seq_lengths, init_state,
                              scope='SecondPath', reuse=False):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warning(scope.name)
                if self.encoder_rnn_type == 'lstm':
                    cell = rnn.BasicLSTMCell(
                        num_units=self.num_lstm_cell_units,
                        state_is_tuple=True)
                elif self.encoder_rnn_type == 'rnn':
                    cell = rnn.BasicRNNCell(num_units=self.num_lstm_cell_units)
                elif self.encoder_rnn_type == 'gru':
                    cell = rnn.GRUCell(num_units=self.num_lstm_cell_units)
                else:
                    raise ValueError('Unknown encoder rnn type')
                new_h, cell_state = tf.nn.dynamic_rnn(
                    cell=cell, inputs=prev_h, sequence_length=seq_lengths,
                    initial_state=init_state, dtype=tf.float32)
                return new_h, cell_state.h, cell_state.c

        # program token [bs, len] -> embedded tokens [len] list of [bs, dim]
        # tensors
        # Embedding
        def Token_Embedding(token_dim, embedding_dim,
                            scope='Token_Embedding', reuse=False):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warning(scope.name)
                # We add token_dim + 1, to use this tokens as a starting token
                # <s>
                embedding_map = tf.get_variable(
                    name="embedding_map", shape=[token_dim + 1, embedding_dim],
                    initializer=tf.random_uniform_initializer(
                        minval=-0.01, maxval=0.01))

                def embedding_lookup(t):
                    embedding = tf.nn.embedding_lookup(embedding_map, t)
                    return embedding
                return embedding_lookup

        # program token feature [bs, u] -> program token [bs, dim_program_token]
        # MLP
        def Token_Decoder(f, token_dim, scope='Token_Decoder', reuse=False):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warning(scope.name)
                _ = fc(f, token_dim, is_train, info=not reuse,
                       batch_norm=False, activation_fn=None, name='fc1')
                return _

        # Per vector encoder
        def Per_Encoder(token_dim, embedding_dim,
                        scope='Per_Encoder', reuse=False):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                def embedding_lookup(t):
                    if not reuse: log.warning(scope.name)
                    _ = fc(t, embedding_dim, is_train,
                           info=not reuse, activation_fn=None, name='fc2')
                    return _
                return embedding_lookup

        # Input {{{
        # =========
        self.ground_truth_program = self.program
        self.gt_tokens = tf.argmax(self.ground_truth_program, axis=1)
        # k list of [bs, ac, max_demo_len - 1] tensor
        self.gt_actions_onehot = [single_a_h
                                  for single_a_h
                                  in tf.unstack(tf.transpose(
                                      self.a_h, [0, 1, 3, 2]), axis=1)]
        # k list of [bs, max_demo_len - 1] tensor
        self.gt_actions_tokens = [single_a_h_token
                                  for single_a_h_token in tf.unstack(
                                      self.a_h_tokens, axis=1)]
        self.gt_per = tf.transpose(self.per, [1, 0, 3, 2])

        def rn_pool(feat, scope='rn_pool', reuse=False):
            # feat: [bs, k, v]
            tile1 = tf.tile(tf.expand_dims(feat, axis=1), [1, self.k, 1, 1])
            tile2 = tf.tile(tf.expand_dims(feat, axis=2), [1, 1, self.k, 1])
            with tf.variable_scope(scope, reuse=reuse):
                bs = self.batch_size
                k = self.k
                _ = tf.reshape(tf.concat([tile1, tile2], axis=3),
                               [bs * k * k, -1])
                _ = fc(_, self.num_lstm_cell_units, is_train,
                       batch_norm=True, name='fc1')
                _ = fc(_, self.num_lstm_cell_units, is_train,
                       batch_norm=True, name='fc2')
                pooled = tf.reduce_mean(
                    tf.reduce_mean(tf.reshape(_, [bs, k, k, -1]), axis=1),
                    axis=1)
            return pooled

        def SummarizeFeature(features, aggregation='avgpool',
                             scope='SummarizeFeature', reuse=False):
            with tf.variable_scope(scope, reuse=reuse):
                # features: [bs, k, v]
                if aggregation == 'avgpool':  # [bs, v]
                    summary = tf.reduce_mean(features, axis=1)
                elif aggregation == 'rn':  # [bs, v]
                    summary = rn_pool(features, reuse=reuse)
                else:
                    raise ValueError('Unknown demo aggregation type')
            return summary

        # a_h = self.a_h
        # }}}

        # Graph {{{
        # =========
        # Demo -> Demo feature
        step1_h_list = []
        step1_c_list = []
        step1_feature_history_list = []
        for i in range(self.k):
            step1_feature_history, step1_h, step1_c = \
                Demo_Encoder(s_h[:, i, :, :, :, :], demo_len[:, i],
                             reuse=i > 0)
            step1_feature_history_list.append(step1_feature_history)
            step1_h_list.append(step1_h)
            step1_c_list.append(step1_c)
        summary_h = SummarizeFeature(tf.stack(step1_h_list, axis=1),
                                     aggregation='avgpool',
                                     scope='summary_h')
        summary_c = SummarizeFeature(tf.stack(step1_c_list, axis=1),
                                     aggregation='avgpool',
                                     scope='summary_c')

        demo_h_list = []
        demo_c_list = []
        demo_feature_history_list = []
        for i in range(self.k):
            demo_feature_history, demo_h, demo_c = \
                SecondPathEncoder(step1_feature_history_list[i],
                                  demo_len[:, i],
                                  rnn.LSTMStateTuple(summary_c, summary_h),
                                  scope='SecondPathEncoder', reuse=i > 0)
            demo_feature_history_list.append(demo_feature_history)
            demo_h_list.append(demo_h)
            demo_c_list.append(demo_c)
        demo_h_summary = SummarizeFeature(tf.stack(demo_h_list, axis=1),
                                          aggregation='rn',
                                          scope='demo_h_summary')
        demo_c_summary = SummarizeFeature(tf.stack(demo_c_list, axis=1),
                                          aggregation='rn',
                                          scope='demo_c_summary')

        def get_DecoderHelper(embedding_lookup, seq_lengths, token_dim,
                              gt_tokens=None, sequence_type='program',
                              unroll_type='teacher_forcing'):
            if unroll_type == 'teacher_forcing' or sequence_type == 'per':
                if gt_tokens is None:
                    raise ValueError('teacher_forcing requires gt_tokens')
                embedding = embedding_lookup(gt_tokens)
                helper = seq2seq.TrainingHelper(embedding, seq_lengths)
            elif unroll_type == 'scheduled_sampling':
                if gt_tokens is None:
                    raise ValueError('scheduled_sampling requires gt_tokens')
                embedding = embedding_lookup(gt_tokens)
                # sample_prob 1.0: always sample from ground truth
                # sample_prob 0.0: always sample from prediction
                helper = seq2seq.ScheduledEmbeddingTrainingHelper(
                    embedding, seq_lengths, embedding_lookup,
                    1.0 - self.sample_prob, seed=None,
                    scheduling_seed=None)
            elif unroll_type == 'greedy':
                # during evaluation, we perform greedy unrolling.
                start_token = tf.zeros([self.batch_size], dtype=tf.int32) + \
                    token_dim
                if sequence_type == 'program':
                    end_token = self.vocab.token2int['m)']
                elif sequence_type == 'action':
                    end_token = token_dim - 1
                else:
                    raise ValueError('Unknown sequence_type')
                helper = seq2seq.GreedyEmbeddingHelper(
                    embedding_lookup, start_token, end_token)
            else:
                raise ValueError('Unknown unroll type')
            return helper

        def LSTM_Decoder(visual_h, visual_c, gt_tokens, lstm_cell,
                         unroll_type='teacher_forcing',
                         seq_lengths=None, max_sequence_len=10, token_dim=50,
                         sequence_type='program', embedding_dim=128,
                         scope='LSTM_Decoder', reuse=False):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                # augmented embedding with token_dim + 1 (<s>) token
                if sequence_type == 'program' or sequence_type == 'action':
                    s_token = tf.zeros([self.batch_size, 1],
                                       dtype=gt_tokens.dtype) + token_dim + 1
                    gt_tokens = tf.concat([s_token, gt_tokens[:, :-1]], axis=1)

                    embedding_lookup = Token_Embedding(token_dim, embedding_dim,
                                                       reuse=reuse)
                else:
                    embedding_lookup = Per_Encoder(token_dim, embedding_dim,
                                                   reuse=reuse)

                # dynamic_decode implementation
                helper = get_DecoderHelper(embedding_lookup, seq_lengths,
                                           token_dim, gt_tokens=gt_tokens,
                                           sequence_type=sequence_type,
                                           unroll_type=unroll_type)
                projection_layer = layer_core.Dense(
                    token_dim, use_bias=False, name="output_projection")
                decoder = seq2seq.BasicDecoder(
                    lstm_cell, helper, rnn.LSTMStateTuple(visual_c, visual_h),
                    output_layer=projection_layer)
                # pred_length [batch_size]: length of the predicted sequence
                outputs, _, pred_length = tf.contrib.seq2seq.dynamic_decode(
                    decoder, maximum_iterations=max_sequence_len,
                    scope='dynamic_decoder')
                pred_length = tf.expand_dims(pred_length, axis=1)

                # as dynamic_decode generate variable length sequence output,
                # we pad it dynamically to match input embedding shape.
                rnn_output = outputs.rnn_output
                sz = tf.shape(rnn_output)
                dynamic_pad = tf.zeros(
                    [sz[0], max_sequence_len - sz[1], sz[2]],
                    dtype=rnn_output.dtype)
                pred_seq = tf.concat([rnn_output, dynamic_pad], axis=1)
                seq_shape = pred_seq.get_shape().as_list()
                pred_seq.set_shape(
                    [seq_shape[0], max_sequence_len, seq_shape[2]])

                pred_seq = tf.transpose(
                    tf.reshape(pred_seq,
                               [self.batch_size, max_sequence_len, -1]),
                    [0, 2, 1])  # make_dim: [bs, n, len]
                return pred_seq, pred_length

        if self.scheduled_sampling:
            train_unroll_type = 'scheduled_sampling'
        else:
            train_unroll_type = 'teacher_forcing'

        # Demo feature -> Program
        self.program_lstm_cell = rnn.BasicLSTMCell(
            num_units=self.num_lstm_cell_units)
        embedding_dim = demo_h_summary.get_shape().as_list()[-1]
        self.pred_program, self.pred_program_len = LSTM_Decoder(
            demo_h_summary, demo_c_summary, self.program_tokens,
            self.program_lstm_cell, unroll_type=train_unroll_type,
            seq_lengths=self.program_len[:, 0],
            max_sequence_len=self.max_program_len,
            token_dim=self.dim_program_token,
            sequence_type='program',
            embedding_dim=embedding_dim, scope='Program_Decoder', reuse=False
        )
        assert self.pred_program.get_shape() == \
            self.ground_truth_program.get_shape()

        self.greedy_pred_program, self.greedy_pred_program_len = LSTM_Decoder(
            demo_h_summary, demo_c_summary, self.program_tokens,
            self.program_lstm_cell, unroll_type='greedy',
            seq_lengths=self.program_len[:, 0],
            max_sequence_len=self.max_program_len,
            token_dim=self.dim_program_token,
            sequence_type='program',
            embedding_dim=embedding_dim, scope='Program_Decoder', reuse=True
        )
        assert self.greedy_pred_program.get_shape() == \
            self.ground_truth_program.get_shape()

        self.action_lstm_cell = rnn.BasicLSTMCell(
            num_units=self.num_lstm_cell_units)
        self.pred_action_list = []
        self.greedy_pred_action_list = []
        self.greedy_pred_action_len_list = []
        for i in range(self.k):
            # pred_action: [bs, token_dim, sequence_length]
            embedding_dim = demo_h_list[i].get_shape().as_list()[-1]
            pred_action, pred_action_len = LSTM_Decoder(
                demo_h_list[i], demo_c_list[i], self.gt_actions_tokens[i],
                self.action_lstm_cell, unroll_type=train_unroll_type,
                seq_lengths=self.action_len[:, i],
                max_sequence_len=self.max_action_len,
                token_dim=self.action_space,
                sequence_type='action',
                embedding_dim=embedding_dim, scope='Action_Decoder',
                reuse=i > 0)
            assert pred_action.get_shape() == \
                self.gt_actions_onehot[i].get_shape()
            self.pred_action_list.append(pred_action)

            greedy_pred_action, greedy_pred_action_len = LSTM_Decoder(
                demo_h_list[i], demo_c_list[i], self.gt_actions_tokens[i],
                self.action_lstm_cell, unroll_type='greedy',
                seq_lengths=self.action_len[:, i],
                max_sequence_len=self.max_action_len,
                token_dim=self.action_space,
                sequence_type='action',
                embedding_dim=embedding_dim, scope='Action_Decoder',
                reuse=True)
            assert greedy_pred_action.get_shape() == \
                self.gt_actions_onehot[i].get_shape()
            self.greedy_pred_action_list.append(greedy_pred_action)
            self.greedy_pred_action_len_list.append(greedy_pred_action_len)
        self.pred_action = tf.transpose(
            tf.stack(self.pred_action_list, axis=0), [1, 0, 3, 2])
        self.greedy_pred_action = tf.transpose(
            tf.stack(self.greedy_pred_action_list, axis=0), [1, 0, 3, 2])

        self.per_lstm_cell = rnn.BasicLSTMCell(
            num_units=self.num_lstm_cell_units)
        self.pred_per_list = []
        self.greedy_pred_per_list = []
        self.greedy_pred_per_len_list = []
        for i in range(self.k):
            # pred_per: [bs, per_dim, sequence_length]
            embedding_dim = demo_h_list[i].get_shape().as_list()[-1]
            pred_per, pred_per_len = LSTM_Decoder(
                demo_h_list[i], demo_c_list[i],
                self.per[:, i],
                self.per_lstm_cell, unroll_type=train_unroll_type,
                seq_lengths=self.action_len[:, i],
                max_sequence_len=self.max_action_len,
                token_dim=self.per_dim,
                sequence_type='per',
                embedding_dim=embedding_dim, scope='Per_Decoder',
                reuse=i > 0)
            self.pred_per_list.append(pred_per)

            greedy_pred_per, greedy_pred_per_len = LSTM_Decoder(
                demo_h_list[i], demo_c_list[i],
                self.per[:, i],
                self.per_lstm_cell, unroll_type='greedy',
                seq_lengths=self.action_len[:, i],
                max_sequence_len=self.max_action_len,
                token_dim=self.per_dim,
                sequence_type='per',
                embedding_dim=embedding_dim, scope='Per_Decoder',
                reuse=True)
            self.greedy_pred_per_list.append(greedy_pred_per)
            self.greedy_pred_per_len_list.append(greedy_pred_per_len)
        self.pred_per = tf.transpose(
            tf.stack(self.pred_per_list, axis=0), [1, 0, 3, 2])
        self.greedy_pred_per = tf.transpose(
            tf.stack(self.greedy_pred_per_list, axis=0), [1, 0, 3, 2])
        # }}}

        def check_correct_syntax(p_token, p_len, is_same_seq):
            if self.dataset_type == 'karel':
                from karel_env.dsl.dsl_parse import parse
            elif self.dataset_type == 'vizdoom':
                from vizdoom_env.dsl.dsl_parse import parse
            is_correct = []
            for i in range(self.batch_size):
                if is_same_seq[i] == 1:
                    is_correct.append(1)
                else:
                    p_str = self.vocab.intseq2str(p_token[i, :p_len[i, 0]])
                    parse_out = parse(p_str)
                    if parse_out[1]: is_correct.append(1)
                    else: is_correct.append(0)
            return np.array(is_correct).astype(np.float32)

        # Build losses {{{
        # ================
        def Sequence_Loss(pred_sequence, gt_sequence,
                          pred_sequence_lengths=None, gt_sequence_lengths=None,
                          max_sequence_len=None, token_dim=None,
                          sequence_type='program',
                          name=None):
            with tf.name_scope(name, "SequenceOutput"):
                max_sequence_lengths = tf.maximum(pred_sequence_lengths,
                                                  gt_sequence_lengths)
                min_sequence_lengths = tf.minimum(pred_sequence_lengths,
                                                  gt_sequence_lengths)
                gt_mask = tf.sequence_mask(gt_sequence_lengths[:, 0],
                                           max_sequence_len, dtype=tf.float32,
                                           name='mask')
                max_mask = tf.sequence_mask(max_sequence_lengths[:, 0],
                                            max_sequence_len, dtype=tf.float32,
                                            name='max_mask')
                min_mask = tf.sequence_mask(min_sequence_lengths[:, 0],
                                            max_sequence_len, dtype=tf.float32,
                                            name='min_mask')
                labels = tf.reshape(tf.transpose(gt_sequence, [0, 2, 1]),
                                    [self.batch_size*max_sequence_len,
                                     token_dim])
                logits = tf.reshape(tf.transpose(pred_sequence, [0, 2, 1]),
                                    [self.batch_size*max_sequence_len,
                                     token_dim])

                # [bs, max_program_len]
                if sequence_type == 'program' or sequence_type == 'action':
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels, logits=logits)
                else:
                    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=labels, logits=logits)
                    cross_entropy = tf.reduce_mean(cross_entropy, axis=-1)

                # normalize loss
                loss = tf.reduce_sum(cross_entropy * tf.reshape(gt_mask, [-1])) / \
                    tf.reduce_sum(gt_mask)
                output = [gt_sequence, pred_sequence]

                if sequence_type == 'program' or sequence_type == 'action':
                    label_argmax = tf.argmax(labels, axis=-1)
                    logit_argmax = tf.argmax(logits, axis=-1)

                    # accuracy
                    # token level acc
                    correct_token_pred = tf.reduce_sum(
                        tf.to_float(tf.equal(label_argmax, logit_argmax)) *
                        tf.reshape(min_mask, [-1]))
                    token_accuracy = correct_token_pred / tf.reduce_sum(max_mask)
                    # seq level acc
                    seq_equal = tf.equal(
                        tf.reshape(tf.to_float(label_argmax) *
                                   tf.reshape(gt_mask, [-1]),
                                   [self.batch_size, -1]),
                        tf.reshape(tf.to_float(logit_argmax) *
                                   tf.reshape(gt_mask, [-1]),
                                   [self.batch_size, -1])
                    )
                    len_equal = tf.equal(gt_sequence_lengths[:, 0],
                                         pred_sequence_lengths[:, 0])
                    is_same_seq = tf.to_float(tf.logical_and(
                                tf.reduce_all(seq_equal, axis=-1), len_equal))
                    seq_accuracy = tf.reduce_sum(is_same_seq) / self.batch_size
                else:
                    token_accuracy = None
                    seq_accuracy = None
                    is_same_seq = None

                if sequence_type == 'program':
                    pred_tokens = tf.reshape(
                        logit_argmax, [self.batch_size, max_sequence_len])
                    is_correct_syntax = tf.py_func(
                        check_correct_syntax,
                        [pred_tokens, pred_sequence_lengths, is_same_seq],
                        tf.float32)
                    syntax_accuracy = \
                        tf.reduce_sum(is_correct_syntax) / self.batch_size
                else:
                    pred_tokens = None
                    syntax_accuracy = None
                    is_correct_syntax = None

                output_stat = SequenceLossOutput(
                    mask=gt_mask, loss=loss, output=output,
                    token_acc=token_accuracy,
                    seq_acc=seq_accuracy, syntax_acc=syntax_accuracy,
                    is_correct_syntax=is_correct_syntax,
                    pred_tokens=pred_tokens, is_same_seq=is_same_seq,
                )

                return output_stat

        def exact_program_compare_karel(p_token, p_len, is_correct_syntax,
                                        gt_token, gt_len):
            from karel_env.dsl import dsl_enum_program

            exact_program_correct = []
            for i in range(self.batch_size):
                if is_correct_syntax[i] == 1:
                    p_str = self.vocab.intseq2str(p_token[i, :p_len[i, 0]])
                    gt_str = self.vocab.intseq2str(gt_token[i, :gt_len[i, 0]])

                    p_prog, _ = dsl_enum_program.parse(p_str)
                    gt_prog, _ = dsl_enum_program.parse(gt_str)
                    exact_program_correct.append(float(p_prog == gt_prog))
                else:
                    exact_program_correct.append(0.0)
            return np.array(exact_program_correct, dtype=np.float32)

        def exact_program_compare_vizdoom(p_token, p_len, is_correct_syntax,
                                          gt_token, gt_len):
            from vizdoom_env.dsl import dsl_enum_program

            exact_program_correct = []
            for i in range(self.batch_size):
                if is_correct_syntax[i] == 1:
                    p_str = self.vocab.intseq2str(p_token[i, :p_len[i, 0]])
                    gt_str = self.vocab.intseq2str(gt_token[i, :gt_len[i, 0]])

                    p_prog, _ = dsl_enum_program.parse(p_str)
                    gt_prog, _ = dsl_enum_program.parse(gt_str)
                    exact_program_correct.append(float(p_prog == gt_prog))
                else:
                    exact_program_correct.append(0.0)
            return np.array(exact_program_correct, dtype=np.float32)

        def generate_program_output_karel(initial_states, max_demo_len,
                                          demo_k, h, w, depth,
                                          p_token, p_len, is_correct_syntax,
                                          is_same_seq):
            from karel_env import karel
            from karel_env.dsl.dsl_parse import parse

            batch_pred_demos = []
            batch_pred_demo_len = []
            for i in range(self.batch_size):
                pred_demos = []
                pred_demo_len = []
                for k in range(demo_k):
                    if is_same_seq[i] == 0 and is_correct_syntax[i] == 1:
                        p_str = self.vocab.intseq2str(p_token[i, :p_len[i, 0]])
                        exe, s_exe = parse(p_str)
                        if not s_exe:
                            raise RuntimeError("s_exe couldn't be False here")
                        karel_world, n, s_run = exe(
                            karel.Karel_world(initial_states[i, k],
                                              make_error=self.env_type != 'no_error'),
                            0)
                        if s_run:
                            exe_s_h = copy.deepcopy(karel_world.s_h)
                            pred_demo_len.append(len(exe_s_h))
                            pred_demo = np.stack(exe_s_h[:pred_demo_len[-1]], axis=0)
                            padded = np.zeros([max_demo_len, h, w, depth])
                            padded[:pred_demo.shape[0], :, :, :] = pred_demo[:max_demo_len]
                            pred_demos.append(padded)
                        else:
                            pred_demo_len.append(0)
                            pred_demos.append(
                                np.zeros([max_demo_len, h, w, depth]))
                    else:
                        pred_demo_len.append(0)
                        pred_demos.append(
                            np.zeros([max_demo_len, h, w, depth]))
                batch_pred_demos.append(np.stack(pred_demos, axis=0))
                batch_pred_demo_len.append(np.stack(pred_demo_len, axis=0))
            return np.stack(batch_pred_demos, axis=0).astype(np.float32), \
                np.stack(batch_pred_demo_len, axis=0).astype(np.int32)

        def generate_program_output_vizdoom(init_pos, init_pos_len,
                                            vizdoom_pos_keys,
                                            max_demo_len,
                                            demo_k, h, w, depth,
                                            p_token, p_len, is_correct_syntax,
                                            is_same_seq):
            from vizdoom_env.vizdoom_env import Vizdoom_env
            from vizdoom_env.dsl.dsl_parse import parse
            from cv2 import resize, INTER_AREA

            world = Vizdoom_env(config='vizdoom_env/asset/default.cfg',
                                perception_type=self.perception_type)
            world.init_game()
            batch_pred_demos = []
            batch_pred_demo_len = []
            for i in range(self.batch_size):
                pred_demos = []
                pred_demo_len = []
                for k in range(demo_k):
                    if is_same_seq[i] == 0 and is_correct_syntax[i] == 1:
                        init_dict = {}
                        for p, key in enumerate(vizdoom_pos_keys):
                            init_dict[key] = np.squeeze(
                                init_pos[i, k, p][:init_pos_len[i, k, p]])
                        world.new_episode(init_dict)
                        p_str = self.vocab.intseq2str(p_token[i, :p_len[i, 0]])
                        exe, compile_sucess = parse(p_str)
                        if not compile_sucess:
                            raise RuntimeError(
                                "Compile failure should not happen here")
                        new_w, num_call, success = exe(world, 0)
                        if success:
                            exe_s_h = []
                            for s in world.s_h:
                                if s.shape[0] != h or s.shape[1] != w:
                                    s = resize(s, (h, w),
                                               interpolation=INTER_AREA)
                                exe_s_h.append(s.copy())
                            pred_demo_len.append(len(exe_s_h))
                            pred_demo = np.stack(exe_s_h[:pred_demo_len[-1]],
                                                 axis=0)
                            padded = np.zeros([max_demo_len, h, w, depth])
                            padded[:pred_demo.shape[0], :, :, :] = \
                                pred_demo[:max_demo_len]
                            pred_demos.append(padded)
                        else:
                            pred_demo_len.append(0)
                            pred_demos.append(
                                np.zeros([max_demo_len, h, w, depth]))
                    else:
                        pred_demo_len.append(0)
                        pred_demos.append(
                            np.zeros([max_demo_len, h, w, depth]))
                batch_pred_demos.append(np.stack(pred_demos, axis=0))
                batch_pred_demo_len.append(np.stack(pred_demo_len, axis=0))
            world.end_game()
            return np.stack(batch_pred_demos, axis=0).astype(np.float32), \
                np.stack(batch_pred_demo_len, axis=0).astype(np.int32)

        def ExecuteProgram(s_h, max_demo_len, k, h, w, depth,
                           p_token, p_len,
                           is_correct_syntax, is_same_seq,
                           init_pos=None,
                           init_pos_len=None):
            if self.dataset_type == 'karel':
                initial_states = s_h[:, :, 0, :, :, :]  # [bs, k, h, w, depth]
                execution, execution_len = tf.py_func(
                    generate_program_output_karel,
                    [initial_states,
                     max_demo_len, k, h, w, depth,
                     p_token, p_len, is_correct_syntax, is_same_seq],
                    (tf.float32, tf.int32))
            elif self.dataset_type == 'vizdoom':
                execution, execution_len = tf.py_func(
                    generate_program_output_vizdoom,
                    [init_pos, init_pos_len, self.vizdoom_pos_keys,
                     max_demo_len, k, h, w, depth,
                     p_token, p_len, is_correct_syntax, is_same_seq],
                    (tf.float32, tf.int32))
            else:
                raise ValueError('Unknown dataset_type')
            execution.set_shape([self.batch_size, k,
                                 max_demo_len, h, w, depth])
            execution_len.set_shape([self.batch_size, k])
            return execution, execution_len

        def ExactProgramCompare(p_token, p_len, is_correct_syntax, gt_token, gt_len):
            if self.dataset_type == 'karel':
                exact_program_correct = tf.py_func(
                    exact_program_compare_karel,
                    [p_token, p_len, is_correct_syntax, gt_token, gt_len],
                    (tf.float32))
            elif self.dataset_type == 'vizdoom':
                exact_program_correct = tf.py_func(
                    exact_program_compare_vizdoom,
                    [p_token, p_len, is_correct_syntax, gt_token, gt_len],
                    (tf.float32))
            else:
                raise ValueError('Unknown dataset_type')
            exact_program_correct.set_shape([self.batch_size])
            exact_program_accuracy = tf.reduce_mean(exact_program_correct)
            return exact_program_correct, exact_program_accuracy

        def CompareDemoAndExecution(demo, demo_len, k,
                                    execution, execution_len,
                                    is_same_program):
            _ = tf.equal(demo, execution)
            _ = tf.reduce_all(_, axis=-1)  # reduce depth
            _ = tf.reduce_all(_, axis=-1)  # reduce w
            _ = tf.reduce_all(_, axis=-1)  # reduce h
            _ = tf.reduce_all(_, axis=-1)  # reduce sequence length
            is_same_execution = _  # [bs, k]
            is_same_len = tf.equal(demo_len, execution_len)  # [bs, k]

            is_correct_execution = tf.logical_or(
                tf.logical_and(is_same_execution, is_same_len),
                tf.tile(
                    tf.expand_dims(tf.cast(is_same_program, tf.bool), axis=1),
                    [1, k]))  # [bs, k]
            num_correct_execution = tf.reduce_sum(
                tf.to_float(is_correct_execution), axis=-1)

            hist_list = []
            for i in range(k + 1):
                eq_i = tf.to_float(tf.equal(num_correct_execution, i))
                hist_list.append(tf.reduce_sum(eq_i) / self.batch_size)
            execution_acc_hist = tf.stack(hist_list, axis=0)
            return num_correct_execution, is_correct_execution, execution_acc_hist

        self.loss = 0
        self.output = []

        program_stat = Sequence_Loss(
            self.pred_program,
            self.ground_truth_program,
            pred_sequence_lengths=self.program_len,
            gt_sequence_lengths=self.program_len,
            max_sequence_len=self.max_program_len,
            token_dim=self.dim_program_token,
            sequence_type='program',
            name="Program_Sequence_Loss")

        self.program_is_correct_syntax = program_stat.is_correct_syntax
        self.loss += program_stat.loss
        self.output.extend(program_stat.output)

        self.pred_exact_program_correct, self.pred_exact_program_accuracy = \
            ExactProgramCompare(program_stat.pred_tokens, self.program_len,
                                program_stat.is_correct_syntax,
                                self.gt_tokens, self.program_len)

        # Execute program with TRAINING demo initial states
        program_execution, program_execution_len = ExecuteProgram(
            self.s_h, self.max_demo_len, self.k, self.h, self.w, self.depth,
            program_stat.pred_tokens, self.program_len,
            program_stat.is_correct_syntax, program_stat.is_same_seq,
            init_pos=self.init_pos, init_pos_len=self.init_pos_len)
        self.program_num_execution_correct, self.program_is_correct_execution, \
            program_execution_acc_hist = \
            CompareDemoAndExecution(self.s_h, self.demo_len, self.k,
                                    program_execution, program_execution_len,
                                    program_stat.is_same_seq)
        # Execute program with TESTING demo initial states
        test_program_execution, test_program_execution_len = ExecuteProgram(
            self.test_s_h, self.max_demo_len,
            self.test_k, self.h, self.w, self.depth,
            program_stat.pred_tokens, self.program_len,
            program_stat.is_correct_syntax, program_stat.is_same_seq,
            init_pos=self.test_init_pos, init_pos_len=self.test_init_pos_len)
        self.test_program_num_execution_correct, \
            self.test_program_is_correct_execution, \
            test_program_execution_acc_hist = \
            CompareDemoAndExecution(self.test_s_h, self.test_demo_len,
                                    self.test_k,
                                    test_program_execution,
                                    test_program_execution_len,
                                    program_stat.is_same_seq)

        greedy_program_stat = Sequence_Loss(
            self.greedy_pred_program,
            self.ground_truth_program,
            pred_sequence_lengths=self.greedy_pred_program_len,
            gt_sequence_lengths=self.program_len,
            max_sequence_len=self.max_program_len,
            token_dim=self.dim_program_token,
            sequence_type='program',
            name="Greedy_Program_Sequence_Loss")

        self.greedy_program_is_correct_syntax = \
            greedy_program_stat.is_correct_syntax

        self.greedy_exact_program_correct, self.greedy_exact_program_accuracy = \
            ExactProgramCompare(greedy_program_stat.pred_tokens, self.greedy_pred_program_len,
                                greedy_program_stat.is_correct_syntax,
                                self.gt_tokens, self.program_len)

        # Execute program with TRAINING demo initial states
        greedy_execution, greedy_execution_len = ExecuteProgram(
            self.s_h, self.max_demo_len, self.k, self.h, self.w, self.depth,
            greedy_program_stat.pred_tokens, self.greedy_pred_program_len,
            greedy_program_stat.is_correct_syntax,
            greedy_program_stat.is_same_seq,
            init_pos=self.init_pos, init_pos_len=self.init_pos_len)
        self.greedy_num_execution_correct, self.greedy_is_correct_execution, \
            greedy_execution_acc_hist = \
            CompareDemoAndExecution(self.s_h, self.demo_len, self.k,
                                    greedy_execution, greedy_execution_len,
                                    greedy_program_stat.is_same_seq)
        # Execute program with TESTING demo initial states
        test_greedy_execution, test_greedy_execution_len = ExecuteProgram(
            self.test_s_h, self.max_demo_len, self.test_k,
            self.h, self.w, self.depth,
            greedy_program_stat.pred_tokens, self.greedy_pred_program_len,
            greedy_program_stat.is_correct_syntax,
            greedy_program_stat.is_same_seq,
            init_pos=self.test_init_pos, init_pos_len=self.test_init_pos_len)
        self.test_greedy_num_execution_correct, \
            self.test_greedy_is_correct_execution, \
            test_greedy_execution_acc_hist = \
            CompareDemoAndExecution(self.test_s_h, self.test_demo_len,
                                    self.test_k,
                                    test_greedy_execution,
                                    test_greedy_execution_len,
                                    greedy_program_stat.is_same_seq)

        action_masks = []
        avg_action_loss = 0
        avg_action_token_acc = 0
        avg_action_seq_acc = 0
        for i in range(self.k):
            action_stat = Sequence_Loss(
                self.pred_action_list[i],
                self.gt_actions_onehot[i],
                pred_sequence_lengths=tf.expand_dims(
                    self.action_len[:, i], axis=1),
                gt_sequence_lengths=tf.expand_dims(
                    self.action_len[:, i], axis=1),
                max_sequence_len=self.max_action_len,
                token_dim=self.action_space,
                sequence_type='action',
                name="Action_Sequence_Loss_{}".format(i))
            action_masks.append(action_stat.mask)
            avg_action_loss += action_stat.loss
            avg_action_token_acc += action_stat.token_acc
            avg_action_seq_acc += action_stat.seq_acc
            self.output.extend(action_stat.output)
        avg_action_loss /= self.k
        avg_action_token_acc /= self.k
        avg_action_seq_acc /= self.k
        self.loss += avg_action_loss

        greedy_avg_action_loss = 0
        greedy_avg_action_token_acc = 0
        greedy_avg_action_seq_acc = 0
        for i in range(self.k):
            greedy_action_stat = Sequence_Loss(
                self.greedy_pred_action_list[i],
                self.gt_actions_onehot[i],
                pred_sequence_lengths=self.greedy_pred_action_len_list[i],
                gt_sequence_lengths=tf.expand_dims(
                    self.action_len[:, i], axis=1),
                max_sequence_len=self.max_action_len,
                token_dim=self.action_space,
                sequence_type='action',
                name="Greedy_Action_Sequence_Loss_{}".format(i))
            greedy_avg_action_loss += greedy_action_stat.loss
            greedy_avg_action_token_acc += greedy_action_stat.token_acc
            greedy_avg_action_seq_acc += greedy_action_stat.seq_acc
        greedy_avg_action_loss /= self.k
        greedy_avg_action_token_acc /= self.k
        greedy_avg_action_seq_acc /= self.k

        per_masks = []
        avg_per_loss = 0
        for i in range(self.k):
            per_stat = Sequence_Loss(
                self.pred_per_list[i],
                self.gt_per[i],
                pred_sequence_lengths=tf.expand_dims(
                    self.action_len[:, i], axis=1),
                gt_sequence_lengths=tf.expand_dims(
                    self.action_len[:, i], axis=1),
                max_sequence_len=self.max_action_len,
                token_dim=self.per_dim,
                sequence_type='per',
                name="Per_Sequence_Loss_{}".format(i))
            per_masks.append(per_stat.mask)
            avg_per_loss += per_stat.loss
            self.output.extend(per_stat.output)
        avg_per_loss /= self.k
        self.loss += avg_per_loss

        greedy_avg_per_loss = 0
        for i in range(self.k):
            greedy_per_stat = Sequence_Loss(
                self.greedy_pred_per_list[i],
                self.gt_per[i],
                pred_sequence_lengths=self.action_len[:, i],
                gt_sequence_lengths=tf.expand_dims(
                    self.action_len[:, i], axis=1),
                max_sequence_len=self.max_action_len,
                token_dim=self.per_dim,
                sequence_type='per',
                name="Greedy_Per_Sequence_Loss_{}".format(i))
            greedy_avg_per_loss += greedy_per_stat.loss
        greedy_avg_per_loss /= self.k
        # }}}

        # Evaluation {{{
        # ==============
        self.report_loss = {}
        self.report_accuracy = {}
        self.report_hist = {}
        self.report_loss['program_loss'] = program_stat.loss
        self.report_accuracy['program_token_acc'] = program_stat.token_acc
        self.report_accuracy['program_seq_acc'] = program_stat.seq_acc
        self.report_accuracy['program_syntax_acc'] = program_stat.syntax_acc
        self.report_accuracy['pred_exact_program_accuracy'] = \
            self.pred_exact_program_accuracy
        self.report_accuracy['greedy_exact_program_accuracy'] = \
            self.greedy_exact_program_accuracy
        self.report_loss['greedy_program_loss'] = greedy_program_stat.loss
        self.report_accuracy['greedy_program_token_acc'] = \
            greedy_program_stat.token_acc
        self.report_accuracy['greedy_program_seq_acc'] = \
            greedy_program_stat.seq_acc
        self.report_accuracy['greedy_program_syntax_acc'] = \
            greedy_program_stat.syntax_acc
        self.report_hist['program_execution_acc_hist'] = \
            program_execution_acc_hist
        self.report_hist['greedy_program_execution_acc_hist'] = \
            greedy_execution_acc_hist
        self.report_hist['test_program_execution_acc_hist'] = \
            test_program_execution_acc_hist
        self.report_hist['test_greedy_program_execution_acc_hist'] = \
            test_greedy_execution_acc_hist
        self.report_loss['avg_action_loss'] = avg_action_loss
        self.report_accuracy['avg_action_token_acc'] = avg_action_token_acc
        self.report_accuracy['avg_action_seq_acc'] = avg_action_seq_acc
        self.report_loss['greedy_avg_action_loss'] = greedy_avg_action_loss
        self.report_accuracy['greedy_avg_action_token_acc'] = \
            greedy_avg_action_token_acc
        self.report_accuracy['greedy_avg_action_seq_acc'] = \
            greedy_avg_action_seq_acc
        self.report_output = []
        #

        # Tensorboard Summary {{{
        # =======================
        # Loss
        def train_test_scalar_summary(name, value):
            tf.summary.scalar(name, value, collections=['train'])
            tf.summary.scalar("test_{}".format(name), value,
                              collections=['test'])

        train_test_scalar_summary("loss/loss", self.loss)
        train_test_scalar_summary("loss/program_loss", program_stat.loss)
        train_test_scalar_summary("loss/program_token_acc",
                                  program_stat.token_acc)
        train_test_scalar_summary("loss/program_seq_acc",
                                  program_stat.seq_acc)
        train_test_scalar_summary("loss/program_syntax_acc",
                                  program_stat.syntax_acc)
        if self.scheduled_sampling:
            train_test_scalar_summary("loss/sample_prob", self.sample_prob)
        tf.summary.scalar("test_loss/greedy_program_loss",
                          greedy_program_stat.loss, collections=['test'])
        tf.summary.scalar("test_loss/greedy_program_token_acc",
                          greedy_program_stat.token_acc, collections=['test'])
        tf.summary.scalar("test_loss/greedy_program_seq_acc",
                          greedy_program_stat.seq_acc, collections=['test'])
        tf.summary.scalar("test_loss/greedy_program_syntax_acc",
                          greedy_program_stat.syntax_acc, collections=['test'])

        train_test_scalar_summary("loss/avg_action_loss", avg_action_loss)
        train_test_scalar_summary(
            "loss/avg_action_token_acc", avg_action_token_acc)
        train_test_scalar_summary(
            "loss/avg_action_seq_acc", avg_action_seq_acc)
        tf.summary.scalar("test_loss/greedy_avg_action_loss",
                          greedy_avg_action_loss, collections=['test'])
        tf.summary.scalar("test_loss/greedy_avg_action_token_acc",
                          greedy_avg_action_token_acc, collections=['test'])
        tf.summary.scalar("test_loss/greedy_avg_action_seq_acc",
                          greedy_avg_action_seq_acc, collections=['test'])

        train_test_scalar_summary("loss/avg_per_loss", avg_per_loss)
        tf.summary.scalar("test_loss/greedy_avg_per_loss",
                          greedy_avg_per_loss, collections=['test'])

        def program2str(p_token, p_len):
            program_str = []
            for i in range(self.batch_size):
                program_str.append(
                    self.vocab.intseq2str(
                        np.argmax(p_token[i], axis=0)[:p_len[i, 0]]))
            program_str = np.stack(program_str, axis=0)
            return program_str

        tf.summary.text('program_id/id', self.program_id, collections=['train'])
        tf.summary.text('program/pred',
                        tf.py_func(
                            program2str,
                            [self.pred_program, self.program_len],
                            tf.string),
                        collections=['train'])
        tf.summary.text('program/ground_truth',
                        tf.py_func(
                            program2str,
                            [self.ground_truth_program, self.program_len],
                            tf.string),
                        collections=['train'])
        tf.summary.text('test_program_id/id', self.program_id,
                        collections=['test'])
        tf.summary.text('test_program/pred',
                        tf.py_func(
                            program2str,
                            [self.pred_program, self.program_len],
                            tf.string),
                        collections=['test'])
        tf.summary.text('test_program/greedy_pred',
                        tf.py_func(
                            program2str,
                            [self.greedy_pred_program,
                             self.greedy_pred_program_len],
                            tf.string),
                        collections=['test'])
        tf.summary.text('test_program/ground_truth',
                        tf.py_func(
                            program2str,
                            [self.ground_truth_program, self.program_len],
                            tf.string),
                        collections=['test'])

        # Visualization
        def visualized_map(pred, gt):
            dummy = tf.expand_dims(tf.zeros_like(pred), axis=-1)
            pred = tf.expand_dims(pred, axis=-1)
            gt = tf.expand_dims(gt, axis=-1)
            return tf.clip_by_value(tf.concat([pred, gt, dummy], axis=-1), 0, 1)

        if self.debug:
            tiled_mask = tf.tile(tf.expand_dims(program_stat.mask, axis=1),
                                 [1, self.dim_program_token, 1])
            tf.summary.image("debug/mask",
                             tf.image.grayscale_to_rgb(
                                 tf.expand_dims(tiled_mask, -1)),
                             collections=['train'])
        tf.summary.image("visualized_program",
                         visualized_map(tf.nn.softmax(self.pred_program, dim=1),
                                        self.ground_truth_program),
                         collections=['train'])
        tf.summary.image("test_visualized_program",
                         visualized_map(tf.nn.softmax(self.pred_program, dim=1),
                                        self.ground_truth_program),
                         collections=['test'])
        tf.summary.image("test_visualized_greedy_program",
                         visualized_map(tf.nn.softmax(self.greedy_pred_program, dim=1),
                                        self.ground_truth_program),
                         collections=['test'])
        if self.dataset_type == 'vizdoom':
            tf.summary.image("state/initial_state",
                             self.s_h[:, 0, 0, :, :, :],
                             collections=['train'])
            tf.summary.image("state/demo_program_1",
                             self.s_h[0, 0, :, :, :, :],
                             max_outputs=self.max_demo_len,
                             collections=['train'])

        i = 0  # show only the first demo (among k)
        if self.debug:
            tiled_mask = tf.tile(tf.expand_dims(action_masks[i], axis=1),
                                 [1, self.action_space, 1])
            tf.summary.image("debug/action_decoder/k_{}/mask".format(i),
                             tf.image.grayscale_to_rgb(
                                 tf.expand_dims(tiled_mask, -1)),
                             collections=['train'])
        tf.summary.image("visualized_action/k_{}".format(i),
                         visualized_map(tf.nn.softmax(self.pred_action_list[i], dim=1),
                                        self.gt_actions_onehot[i]),
                         collections=['train'])
        tf.summary.image("test_visualized_action/k_{}".format(i),
                         visualized_map(tf.nn.softmax(self.pred_action_list[i], dim=1),
                                        self.gt_actions_onehot[i]),
                         collections=['test'])
        tf.summary.image("test_visualized_greedy_action/k_{}".format(i),
                         visualized_map(tf.nn.softmax(self.greedy_pred_action_list[i], dim=1),
                                        self.gt_actions_onehot[i]),
                         collections=['test'])

        tf.summary.image("visualized_perception",
                         tf.transpose(
                             visualized_map(tf.nn.sigmoid(self.pred_per[:, 0]),
                                            self.per[:, 0]), [0, 2, 1, 3]),
                         collections=['train'])
        tf.summary.image("test_visualized_perception",
                         tf.transpose(
                             visualized_map(tf.nn.sigmoid(self.pred_per[:, 0]),
                                            self.per[:, 0]), [0, 2, 1, 3]),
                         collections=['test'])

        # Visualize demo features
        if self.debug:
            i = 0  # show only the first images
            tf.summary.image("debug/demo_feature_history/k_{}".format(i),
                             tf.image.grayscale_to_rgb(
                                 tf.expand_dims(demo_feature_history_list[i],
                                                -1)),
                             collections=['train'])
        # }}}
        print('\033[93mSuccessfully loaded the model.\033[0m')
