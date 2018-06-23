from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.framework import ops
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from collections import namedtuple
from models.util import log
from models.ops import fc, conv2d


SequenceLossOutput = namedtuple(
    'SequenceLossOutput',
    'mask loss output token_acc seq_acc syntax_acc ' +
    'is_correct_syntax pred_tokens is_same_seq')


def _compute_attention(attention_mechanism, cell_output, previous_alignments,
                       attention_layer, reuse=False):
    """Computes the attention and alignments for a given attention_mechanism."""
    with tf.variable_scope('compute_attention', reuse=reuse):
        alignments = attention_mechanism(
            cell_output, previous_alignments=previous_alignments)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = array_ops.expand_dims(alignments, 1)
    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #   [batch_size, memory_time, attention_mechanism.num_units]
    # the batched matmul is over memory_time, so the output shape is
    #   [batch_size, 1, attention_mechanism.num_units].
    # we then squeeze out the singleton dim.
    context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
    context = array_ops.squeeze(context, [1])

    if attention_layer is not None:
        with tf.variable_scope('compute_attention', reuse=reuse):
            attention = attention_layer(array_ops.concat([cell_output, context], 1))
    else:
        attention = context

    return attention, alignments


class PoolingAttentionWrapper(seq2seq.AttentionWrapper):
    """AttentionWrapper with multiple attention aggregation based on pooling.
    """
    def __init__(self,
                 cell,
                 attention_mechanism,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 pooling='avgpool',
                 name=None):
        """Construct the `PoolingAttentionWrapper`.

        Args:
            attention_layer_size: single Python integer.


        Otherwise, this is same as the AtentionWrapper.
        """
        super(PoolingAttentionWrapper, self).__init__(
            cell,
            attention_mechanism,
            attention_layer_size=None,
            alignment_history=alignment_history,
            cell_input_fn=cell_input_fn,
            output_attention=output_attention,
            initial_cell_state=initial_cell_state,
            name=name)
        self.pooling = pooling

        if attention_layer_size is not None:
            if not isinstance(attention_layer_size, (int, float)):
                raise ValueError('attention_layer_size should be a single number.')
            self._attention_layers = tuple(
                layers_core.Dense(
                    attention_layer_size, name="attention_layer", use_bias=False)
                for _ in range(len(self._attention_mechanisms)))
            self._attention_layer_size = attention_layer_size
        else:
            attention_layer_sizes = [
                attention_mechanism.values.get_shape()[-1].value
                for attention_mechanism in self._attention_mechanisms]
            if max(attention_layer_sizes) != min(attention_layer_sizes):
                raise ValueError('When attention_layer_size is None, '
                                 'attention mechanisms should output with the '
                                 'same dimension.')
            self._attention_layers = None
            self._attention_layer_size = min(attention_layer_sizes)

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.

        - Step 1: Mix the `inputs` and previous step's `attention` output via
          `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
          `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
          alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by POOLING the cell output
          and context through the attention layer (a linear layer with
          `attention_layer_size` outputs).
        """
        if not isinstance(state, seq2seq.AttentionWrapperState):
            raise TypeError("Expected state to be instance of "
                            "AttentionWrapperState. "
                            "Received type %s instead." % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
            cell_output.shape[0].value or array_ops.shape(cell_output)[0])
        error_message = (
            "When applying AttentionWrapper %s: " % self.name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input via "
            "the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = array_ops.identity(
                cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_alignments = state.alignments
            previous_alignment_history = state.alignment_history
        else:
            previous_alignments = [state.alignments]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_histories = []
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments = _compute_attention(
                attention_mechanism, cell_output, previous_alignments[i],
                self._attention_layers[i] if self._attention_layers else None,
                reuse=i > 0)
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            all_alignments.append(alignments)
            all_histories.append(alignment_history)
            all_attentions.append(attention)

        if self.pooling == 'avgpool':
            attention = tf.reduce_mean(tf.stack(all_attentions, axis=1), axis=1)
        else:
            raise ValueError('Unknown pooling method')
        next_state = seq2seq.AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(all_histories))

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state


class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True, global_step=None):
        self.debug = debug_information
        self.global_step = global_step

        self.config = config
        self.pixel_input = self.config.pixel_input or self.config.dataset_type == 'vizdoom'
        self.attn_type = self.config.attn_type
        self.scheduled_sampling = \
            getattr(self.config, 'scheduled_sampling', False) or False
        self.scheduled_sampling_decay_steps = \
            getattr(self.config, 'scheduled_sampling_decay_steps', 5000) or 5000
        self.batch_size = self.config.batch_size
        self.state_encoder_fc = self.config.state_encoder_fc
        self.concat_state_feature_direct_prediction = \
            self.config.concat_state_feature_direct_prediction
        self.encoder_rnn_type = self.config.encoder_rnn_type
        self.dataset_type = self.config.dataset_type
        self.dsl_type = self.config.dsl_type
        self.env_type = self.config.env_type
        self.vizdoom_pos_keys = self.config.vizdoom_pos_keys
        self.vizdoom_max_init_pos_len = self.config.vizdoom_max_init_pos_len
        self.perception_type = self.config.perception_type
        self.level = self.config.level
        self.stack_subsequent_state = self.config.stack_subsequent_state
        self.num_lstm_cell_units = self.config.num_lstm_cell_units
        self.demo_aggregation = self.config.demo_aggregation
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
            from karel.dsl import get_KarelDSL
            self.vocab = get_KarelDSL(dsl_type=self.dsl_type, seed=123)
        else:
            from vizdoom_world.dsl.vocab import VizDoomDSLVocab
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
            shape=[self.batch_size, self.test_k, self.max_demo_len,
                   self.h, self.w, self.depth],
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

        self.test_a_h = tf.placeholder(
            name='test_a_h', dtype=tf.float32,
            shape=[self.batch_size, self.test_k, self.max_demo_len,
                   self.action_space],
        )

        self.test_a_h_tokens = tf.placeholder(
            name='test_a_h_tokens', dtype=tf.int32,
            shape=[self.batch_size, self.test_k, self.max_action_len],
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
        self.action_len = self.demo_len

        self.test_demo_len = tf.placeholder(
            name='test_demo_len', dtype=tf.float32,
            shape=[self.batch_size, self.test_k],
        )
        self.test_demo_len = tf.cast(self.test_demo_len, dtype=tf.int32)
        self.test_action_len = self.test_demo_len

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
            self.test_s_h: batch_chunk['test_s_h'],
            self.a_h: batch_chunk['a_h'],
            self.a_h_tokens: batch_chunk['a_h_tokens'],
            self.test_a_h: batch_chunk['test_a_h'],
            self.test_a_h_tokens: batch_chunk['test_a_h_tokens'],
            self.program_len: batch_chunk['program_len'],
            self.demo_len: batch_chunk['demo_len'],
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
        # demo_len = self.demo_len
        if self.stack_subsequent_state:
            max_demo_len = self.max_demo_len - 1
            demo_len = self.demo_len - 1
            s_h = tf.stack([self.s_h[:, :, :max_demo_len, :, :, :],
                            self.s_h[:, :, 1:, :, :, :]], axis=-1)
            depth = self.depth * 2
        else:
            max_demo_len = self.max_demo_len
            demo_len = self.demo_len
            s_h = self.s_h
            depth = self.depth

        # s [bs, h, w, depth] -> feature [bs, v]
        # CNN
        def State_Encoder(s, per, batch_size, scope='State_Encoder', reuse=False):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warning(scope.name)
                _ = conv2d(s, 16, is_train, k_h=3, k_w=3,
                           info=not reuse, batch_norm=True, name='conv1')
                _ = conv2d(_, 32, is_train, k_h=3, k_w=3,
                           info=not reuse, batch_norm=True, name='conv2')
                _ = conv2d(_, 48, is_train, k_h=3, k_w=3,
                           info=not reuse, batch_norm=True, name='conv3')
                if self.pixel_input:
                    _ = conv2d(_, 48, is_train, k_h=3, k_w=3,
                               info=not reuse, batch_norm=True, name='conv4')
                    _ = conv2d(_, 48, is_train, k_h=3, k_w=3,
                               info=not reuse, batch_norm=True, name='conv5')
                state_feature = tf.reshape(_, [batch_size, -1])
                if self.state_encoder_fc:
                    state_feature = fc(state_feature, 512, is_train,
                                       info=not reuse, name='fc1')
                    state_feature = fc(state_feature, 512, is_train,
                                       info=not reuse, name='fc2')
                state_feature = tf.concat([state_feature, per], axis=-1)
                if not reuse: log.info(
                    'concat feature {}'.format(state_feature))
                return state_feature

        # s_h [bs, t, h, w, depth] -> feature [bs, v]
        # LSTM
        def Demo_Encoder(s_h, per, seq_lengths, scope='Demo_Encoder', reuse=False):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warning(scope.name)
                state_features = tf.reshape(
                    State_Encoder(tf.reshape(s_h, [-1, self.h, self.w, depth]),
                                  tf.reshape(per, [-1, self.per_dim]),
                                  self.batch_size * max_demo_len, reuse=reuse),
                    [self.batch_size, max_demo_len, -1])
                if self.encoder_rnn_type == 'bilstm':
                    fcell = rnn.BasicLSTMCell(
                        num_units=math.ceil(self.num_lstm_cell_units),
                        state_is_tuple=True)
                    bcell = rnn.BasicLSTMCell(
                        num_units=math.floor(self.num_lstm_cell_units),
                        state_is_tuple=True)
                    new_h, cell_state = tf.nn.bidirectional_dynamic_rnn(
                        fcell, bcell, state_features,
                        sequence_length=seq_lengths, dtype=tf.float32)
                    new_h = tf.reduce_sum(tf.stack(new_h, axis=2), axis=2)
                    cell_state = rnn.LSTMStateTuple(
                        tf.reduce_sum(tf.stack(
                            [cs.c for cs in cell_state], axis=1), axis=1),
                        tf.reduce_sum(tf.stack(
                            [cs.h for cs in cell_state], axis=1), axis=1))
                elif self.encoder_rnn_type == 'lstm':
                    cell = rnn.BasicLSTMCell(
                        num_units=self.num_lstm_cell_units,
                        state_is_tuple=True)
                    new_h, cell_state = tf.nn.dynamic_rnn(
                        cell=cell, dtype=tf.float32, sequence_length=seq_lengths,
                        inputs=state_features)
                elif self.encoder_rnn_type == 'rnn':
                    cell = rnn.BasicRNNCell(num_units=self.num_lstm_cell_units)
                    new_h, cell_state = tf.nn.dynamic_rnn(
                        cell=cell, dtype=tf.float32, sequence_length=seq_lengths,
                        inputs=state_features)
                elif self.encoder_rnn_type == 'gru':
                    cell = rnn.GRUCell(num_units=self.num_lstm_cell_units)
                    new_h, cell_state = tf.nn.dynamic_rnn(
                        cell=cell, dtype=tf.float32, sequence_length=seq_lengths,
                        inputs=state_features)
                else:
                    raise ValueError('Unknown encoder rnn type')

                if self.concat_state_feature_direct_prediction:
                    all_states = tf.concat([new_h, state_features], axis=-1)
                else:
                    all_states = new_h
                return all_states, cell_state.h, cell_state.c

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

        # Input {{{
        # =========
        # test_k list of [bs, ac, max_demo_len - 1] tensor
        self.gt_test_actions_onehot = [
            single_test_a_h for single_test_a_h in tf.unstack(tf.transpose(
                self.test_a_h, [0, 1, 3, 2]), axis=1)]
        # test_k list of [bs, max_demo_len - 1] tensor
        self.gt_test_actions_tokens = [
            single_test_a_h_token for single_test_a_h_token in tf.unstack(
                self.test_a_h_tokens, axis=1)]

        # a_h = self.a_h
        # }}}

        # Graph {{{
        # =========
        # Demo -> Demo feature
        demo_h_list = []
        demo_c_list = []
        demo_feature_history_list = []
        for i in range(self.k):
            demo_feature_history, demo_h, demo_c = \
                Demo_Encoder(s_h[:, i], self.per[:, i],
                             demo_len[:, i], reuse=i > 0)
            demo_feature_history_list.append(demo_feature_history)
            demo_h_list.append(demo_h)
            demo_c_list.append(demo_c)
            if i == 0: log.warning(demo_feature_history)
        demo_h_stack = tf.stack(demo_h_list, axis=1)  # [bs, k, v]
        demo_c_stack = tf.stack(demo_c_list, axis=1)  # [bs, k, v]
        if self.demo_aggregation == 'concat':  # [bs, k*v]
            demo_h_summary = tf.reshape(demo_h_stack, [self.batch_size, -1])
            demo_c_summary = tf.reshape(demo_c_stack, [self.batch_size, -1])
        elif self.demo_aggregation == 'avgpool':  # [bs, v]
            demo_h_summary = tf.reduce_mean(demo_h_stack, axis=1)
            demo_c_summary = tf.reduce_mean(demo_c_stack, axis=1)
        elif self.demo_aggregation == 'maxpool':  # [bs, v]
            demo_h_summary = tf.squeeze(
                tf.layers.max_pooling1d(demo_h_stack,
                                        demo_h_stack.get_shape().as_list()[1],
                                        1, padding='valid',
                                        data_format='channels_last'),
                axis=1)
            demo_c_summary = tf.squeeze(
                tf.layers.max_pooling1d(demo_c_stack,
                                        demo_c_stack.get_shape().as_list()[1],
                                        1, padding='valid',
                                        data_format='channels_last'),
                axis=1)
        else:
            raise ValueError('Unknown demo aggregation type')

        def get_DecoderHelper(embedding_lookup, seq_lengths, token_dim,
                              gt_tokens=None, unroll_type='teacher_forcing'):
            if unroll_type == 'teacher_forcing':
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
                start_token = tf.zeros([self.batch_size], dtype=tf.int32) + token_dim
                end_token = token_dim - 1
                helper = seq2seq.GreedyEmbeddingHelper(
                    embedding_lookup, start_token, end_token)
            else:
                raise ValueError('Unknown unroll type')
            return helper

        def LSTM_Decoder(visual_h, visual_c, gt_tokens, lstm_cell,
                         unroll_type='teacher_forcing',
                         seq_lengths=None, max_sequence_len=10, token_dim=50,
                         embedding_dim=128, init_state=None,
                         scope='LSTM_Decoder', reuse=False):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warning(scope.name)
                # augmented embedding with token_dim + 1 (<s>) token
                s_token = tf.zeros([self.batch_size, 1],
                                   dtype=gt_tokens.dtype) + token_dim + 1
                gt_tokens = tf.concat([s_token, gt_tokens[:, :-1]], axis=1)

                embedding_lookup = Token_Embedding(token_dim, embedding_dim,
                                                   reuse=reuse)

                # dynamic_decode implementation
                helper = get_DecoderHelper(embedding_lookup, seq_lengths,
                                           token_dim, gt_tokens=gt_tokens,
                                           unroll_type=unroll_type)
                projection_layer = layers_core.Dense(
                    token_dim, use_bias=False, name="output_projection")
                if init_state is None:
                    init_state = rnn.LSTMStateTuple(visual_c, visual_h)
                decoder = seq2seq.BasicDecoder(
                    lstm_cell, helper, init_state,
                    output_layer=projection_layer)
                # pred_length [batch_size]: length of the predicted sequence
                outputs, final_context_state, pred_length = seq2seq.dynamic_decode(
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
                return pred_seq, pred_length, final_context_state

        if self.scheduled_sampling:
            train_unroll_type = 'scheduled_sampling'
        else:
            train_unroll_type = 'teacher_forcing'

        # Attn
        lstm_cell = rnn.BasicLSTMCell(
            num_units=self.num_lstm_cell_units)
        attn_mechanisms = []
        for j in range(self.test_k):
            attn_mechanisms_k = []
            for i in range(self.k):
                with tf.variable_scope('AttnMechanism', reuse=i > 0 or j > 0):
                    if self.attn_type == 'luong':
                        attn_mechanism = seq2seq.LuongAttention(
                            self.num_lstm_cell_units, demo_feature_history_list[i],
                            memory_sequence_length=self.demo_len[:, i])
                    elif self.attn_type == 'luong_monotonic':
                        attn_mechanism = seq2seq.LuongMonotonicAttention(
                            self.num_lstm_cell_units, demo_feature_history_list[i],
                            memory_sequence_length=self.demo_len[:, i])
                    else:
                        raise ValueError('Unknown attention type')
                attn_mechanisms_k.append(attn_mechanism)
            attn_mechanisms.append(attn_mechanisms_k)

        self.attn_cells = []
        for i in range(self.test_k):
            attn_cell = PoolingAttentionWrapper(
                lstm_cell, attn_mechanisms[i],
                attention_layer_size=self.num_lstm_cell_units,
                alignment_history=True,
                output_attention=True,
                pooling='avgpool')
            self.attn_cells.append(attn_cell)

        # Demo + current state -> action
        self.pred_action_list = []
        self.greedy_pred_action_list = []
        self.greedy_pred_action_len_list = []
        for i in range(self.test_k):
            attn_init_state = self.attn_cells[i].zero_state(
                self.batch_size, dtype=tf.float32).clone(
                    cell_state=rnn.LSTMStateTuple(demo_h_summary, demo_c_summary))
            embedding_dim = demo_h_summary.get_shape().as_list()[-1]
            pred_action, pred_action_len, action_state = LSTM_Decoder(
                    demo_h_summary, demo_c_summary, self.gt_test_actions_tokens[i],
                    self.attn_cells[i], unroll_type=train_unroll_type,
                    seq_lengths=self.test_action_len[:, i],
                    max_sequence_len=self.max_action_len,
                    token_dim=self.action_space,
                    embedding_dim=embedding_dim,
                    init_state=attn_init_state,
                    scope='Manipulation', reuse=i > 0
                )
            assert pred_action.get_shape() == \
                self.gt_test_actions_onehot[i].get_shape()
            self.pred_action_list.append(pred_action)

            greedy_attn_init_state = self.attn_cells[i].zero_state(
                self.batch_size, dtype=tf.float32).clone(
                    cell_state=rnn.LSTMStateTuple(demo_h_summary, demo_c_summary))
            greedy_pred_action, greedy_pred_action_len, \
                greedy_action_state = LSTM_Decoder(
                    demo_h_summary, demo_c_summary, self.gt_test_actions_tokens[i],
                    self.attn_cells[i], unroll_type='greedy',
                    seq_lengths=self.test_action_len[:, i],
                    max_sequence_len=self.max_action_len,
                    token_dim=self.action_space,
                    embedding_dim=embedding_dim,
                    init_state=greedy_attn_init_state,
                    scope='Manipulation', reuse=True
                )
            assert greedy_pred_action.get_shape() == \
                self.gt_test_actions_onehot[i].get_shape()
            self.greedy_pred_action_list.append(greedy_pred_action)
            self.greedy_pred_action_len_list.append(greedy_pred_action_len)
        # }}}

        # Build losses {{{
        # ================
        def Sequence_Loss(pred_sequence, gt_sequence,
                          pred_sequence_lengths=None, gt_sequence_lengths=None,
                          max_sequence_len=None, token_dim=None,
                          sequence_type='program',
                          name=None):
            with tf.name_scope(name, "SequenceOutput") as scope:
                log.warning(scope)
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
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits)
                # normalize loss
                loss = tf.reduce_sum(cross_entropy * tf.reshape(gt_mask, [-1])) / \
                    tf.reduce_sum(gt_mask)
                output = [gt_sequence, pred_sequence]

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
                is_same_seq = tf.logical_and(
                            tf.reduce_all(seq_equal, axis=-1), len_equal)
                seq_accuracy = tf.reduce_sum(
                    tf.to_float(is_same_seq)) / self.batch_size

                pred_tokens = None
                syntax_accuracy = None
                is_correct_syntax = None

                output_stat = SequenceLossOutput(
                    mask=gt_mask, loss=loss, output=output,
                    token_acc=token_accuracy,
                    seq_acc=seq_accuracy,
                    syntax_acc=syntax_accuracy,
                    is_correct_syntax=is_correct_syntax,
                    pred_tokens=pred_tokens, is_same_seq=is_same_seq,
                )

                return output_stat

        self.loss = 0
        self.output = []

        # Manipulation network loss
        avg_action_loss = 0
        avg_action_token_acc = 0
        avg_action_seq_acc = 0
        seq_match = []
        for i in range(self.test_k):
            action_stat = Sequence_Loss(
                self.pred_action_list[i],
                self.gt_test_actions_onehot[i],
                pred_sequence_lengths=tf.expand_dims(
                    self.test_action_len[:, i], axis=1),
                gt_sequence_lengths=tf.expand_dims(
                    self.test_action_len[:, i], axis=1),
                max_sequence_len=self.max_action_len,
                token_dim=self.action_space,
                sequence_type='action',
                name="Action_Sequence_Loss_{}".format(i))
            avg_action_loss += action_stat.loss
            avg_action_token_acc += action_stat.token_acc
            avg_action_seq_acc += action_stat.seq_acc
            seq_match.append(action_stat.is_same_seq)
            self.output.extend(action_stat.output)
        avg_action_loss /= self.test_k
        avg_action_token_acc /= self.test_k
        avg_action_seq_acc /= self.test_k
        avg_action_seq_all_acc = tf.reduce_sum(
            tf.to_float(tf.reduce_all(tf.stack(seq_match, axis=1), axis=-1))
        )/self.batch_size
        self.loss += avg_action_loss

        greedy_avg_action_loss = 0
        greedy_avg_action_token_acc = 0
        greedy_avg_action_seq_acc = 0
        greedy_seq_match = []
        for i in range(self.test_k):
            greedy_action_stat = Sequence_Loss(
                self.greedy_pred_action_list[i],
                self.gt_test_actions_onehot[i],
                pred_sequence_lengths=self.greedy_pred_action_len_list[i],
                gt_sequence_lengths=tf.expand_dims(
                    self.test_action_len[:, i], axis=1),
                max_sequence_len=self.max_action_len,
                token_dim=self.action_space,
                sequence_type='action',
                name="Greedy_Action_Sequence_Loss_{}".format(i))
            greedy_avg_action_loss += greedy_action_stat.loss
            greedy_avg_action_token_acc += greedy_action_stat.token_acc
            greedy_avg_action_seq_acc += greedy_action_stat.seq_acc
            greedy_seq_match.append(greedy_action_stat.is_same_seq)
        greedy_avg_action_loss /= self.test_k
        greedy_avg_action_token_acc /= self.test_k
        greedy_avg_action_seq_acc /= self.test_k
        greedy_avg_action_seq_all_acc = tf.reduce_sum(
            tf.to_float(tf.reduce_all(tf.stack(greedy_seq_match, axis=1), axis=-1))
        )/self.batch_size
        # }}}

        # Evalutaion {{{
        # ==============
        self.report_loss = {}
        self.report_accuracy = {}
        self.report_hist = {}
        self.report_loss['avg_action_loss'] = avg_action_loss
        self.report_accuracy['avg_action_token_acc'] = avg_action_token_acc
        self.report_accuracy['avg_action_seq_acc'] = avg_action_seq_acc
        self.report_accuracy['avg_action_seq_all_acc'] = avg_action_seq_all_acc
        self.report_loss['greedy_avg_action_loss'] = greedy_avg_action_loss
        self.report_accuracy['greedy_avg_action_token_acc'] = \
            greedy_avg_action_token_acc
        self.report_accuracy['greedy_avg_action_seq_acc'] = \
            greedy_avg_action_seq_acc
        self.report_accuracy['greedy_avg_action_seq_all_acc'] = \
            greedy_avg_action_seq_all_acc
        self.report_output = []
        # dummy fetch values for evaler
        self.ground_truth_program = self.program
        self.pred_program = []
        self.greedy_pred_program = []
        self.greedy_pred_program_len = []
        self.greedy_program_is_correct_syntax = []
        self.program_is_correct_syntax = []
        self.program_num_execution_correct = []
        self.program_is_correct_execution = []
        self.greedy_num_execution_correct = []
        self.greedy_is_correct_execution = []
        #

        # Tensorboard Summary {{{
        # =======================
        # Loss
        def train_test_scalar_summary(name, value):
            tf.summary.scalar(name, value, collections=['train'])
            tf.summary.scalar("test_{}".format(name), value,
                              collections=['test'])

        train_test_scalar_summary("loss/loss", self.loss)

        if self.scheduled_sampling:
            train_test_scalar_summary("loss/sample_prob", self.sample_prob)
        train_test_scalar_summary("loss/avg_action_loss", avg_action_loss)
        train_test_scalar_summary("loss/avg_action_token_acc",
                                  avg_action_token_acc)
        train_test_scalar_summary("loss/avg_action_seq_acc",
                                  avg_action_seq_acc)
        train_test_scalar_summary("loss/avg_action_seq_all_acc",
                                  avg_action_seq_all_acc)
        tf.summary.scalar("test_loss/greedy_avg_action_loss",
                          greedy_avg_action_loss, collections=['test'])
        tf.summary.scalar("test_loss/greedy_avg_action_token_acc",
                          greedy_avg_action_token_acc, collections=['test'])
        tf.summary.scalar("test_loss/greedy_avg_action_seq_acc",
                          greedy_avg_action_seq_acc, collections=['test'])
        tf.summary.scalar("test_loss/greedy_avg_action_seq_all_acc",
                          greedy_avg_action_seq_all_acc, collections=['test'])

        def program2str(p_token, p_len):
            program_str = []
            for i in range(self.batch_size):
                program_str.append(
                    self.vocab.intseq2str(
                        np.argmax(p_token[i], axis=0)[:p_len[i, 0]]))
            program_str = np.stack(program_str, axis=0)
            return program_str

        tf.summary.text('program_id/id', self.program_id, collections=['train'])
        tf.summary.text('program/ground_truth',
                        tf.py_func(
                            program2str,
                            [self.program, self.program_len],
                            tf.string),
                        collections=['train'])
        tf.summary.text('test_program_id/id', self.program_id,
                        collections=['test'])
        tf.summary.text('test_program/ground_truth',
                        tf.py_func(
                            program2str,
                            [self.program, self.program_len],
                            tf.string),
                        collections=['test'])

        # Visualization
        def visualized_map(pred, gt):
            dummy = tf.expand_dims(tf.zeros_like(pred), axis=-1)
            pred = tf.expand_dims(tf.nn.softmax(pred, dim=1), axis=-1)
            gt = tf.expand_dims(gt, axis=-1)
            return tf.concat([pred, gt, dummy], axis=-1)

        # Attention visualization
        def build_alignments(alignment_history):
            alignments = []
            for i in alignment_history:
                align = tf.expand_dims(
                    tf.transpose(i.stack(), [1, 2, 0]), -1) * 255
                align_shape = tf.shape(align)
                alignments.append(align)
                alignments.append(tf.zeros(
                    [align_shape[0], 1, align_shape[2], 1],
                    dtype=tf.float32) + 255)
            alignments_image = tf.reshape(
                tf.tile(tf.concat(alignments, axis=1),
                        [1, 1, 1, self.k]),
                [align_shape[0], -1, align_shape[2] * self.k, 1])
            return alignments_image

        alignments = build_alignments(action_state.alignment_history)
        tf.summary.image("attn", alignments, collections=['train'])
        tf.summary.image("test_attn", alignments, collections=['test'])

        greedy_alignments = build_alignments(greedy_action_state.alignment_history)
        tf.summary.image("test_greedy_attn", greedy_alignments, collections=['test'])

        if self.pixel_input:
            tf.summary.image("state/initial_state",
                             self.s_h[:, 0, 0, :, :, :],
                             collections=['train'])
            tf.summary.image("state/demo_program_1",
                             self.s_h[0, 0, :, :, :, :],
                             max_outputs=self.max_demo_len,
                             collections=['train'])

        i = 0  # show only the first demo (among k)
        tf.summary.image("visualized_action/k_{}".format(i),
                         visualized_map(self.pred_action_list[i],
                                        self.gt_test_actions_onehot[i]),
                         collections=['train'])
        tf.summary.image("test_visualized_action/k_{}".format(i),
                         visualized_map(self.pred_action_list[i],
                                        self.gt_test_actions_onehot[i]),
                         collections=['test'])
        tf.summary.image("test_visualized_greedy_action/k_{}".format(i),
                         visualized_map(self.greedy_pred_action_list[i],
                                        self.gt_test_actions_onehot[i]),
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
