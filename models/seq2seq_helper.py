import collections
import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import core as layer_core
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape


class BasicVectorDecoderOutput(
    collections.namedtuple("BasicVectorDecoderOutput",
                           ("rnn_output", "stop_vector", "stop_id"))):
  pass


class BasicVectorDecoder(seq2seq.BasicDecoder):
    """Basic sampling decoder with vector input / output."""

    def __init__(self, cell, helper, initial_state, output_layer=None):
        """Initialize BasicVectorDecoder.

        Args:
            cell: An `RNNCell` instance.
            helper: A `Helper` instance.
            initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
                The initial state of the RNNCell.
            output_layer:An instance of `tf.layers.Layer`, i.e., `tf.layers.Dense`.
                If not provided, use 1 fc layer.

        Raises:
            TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
        """
        if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
            raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
        if not isinstance(helper, helper_py.Helper):
            raise TypeError("helper must be a Helper, received: %s" % type(helper))
        if (output_layer is not None and not isinstance(output_layer, layers_base.Layer)):
            raise TypeError(
                "output_layer must be a Layer, received: %s" % type(output_layer))
        self._cell = cell
        self._helper = helper
        self._initial_state = initial_state
        if output_layer is None:
            self._output_layer = layer_core.Dense(2, use_bias=True,
                                                  name="stop_predictor")
        else:
            self._output_layer = output_layer

    def _rnn_output_size(self):
        return self._cell.output_size

    @property
    def output_size(self):
        # Return the cell output and the id
        return BasicVectorDecoderOutput(
            rnn_output=self._rnn_output_size(),
            # stop_vector=tensor_shape.TensorShape([self.batch_size, 2]),
            stop_vector=2,
            stop_id=tensor_shape.TensorShape([]))

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and int32 (the id)
        dtype = nest.flatten(self._initial_state)[0].dtype
        return BasicVectorDecoderOutput(
            nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            dtype, dtypes.int32)

    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.

        Args:
            time: scalar `int32` tensor.
            inputs: A (structure of) input tensors.
            state: A (structure of) state tensors and TensorArrays.
            name: Name scope for any created operations.

        Returns:
            `(outputs, next_state, next_inputs, finished)`.
        """
        with ops.name_scope(name, "BasicVectorDecoderStep", (time, inputs, state)):
            cell_outputs, cell_state = self._cell(inputs, state)
            stop_vector = self._output_layer(cell_outputs)
            stop_id = self._helper.sample(
                time=time, stop_vector=stop_vector, state=cell_state)
            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                state=cell_state,
                stop_id=stop_id)
        outputs = BasicVectorDecoderOutput(cell_outputs,
                                           stop_vector,
                                           stop_id)
        return (outputs, next_state, next_inputs, finished)


class VectorTrainingHelper(seq2seq.TrainingHelper):
    """A helper for use during training. Only reads inputs.

    Contrary to TrainingHelper, this reads vector inputs.
    stop_id is sampled in a greedy fashion.
    """
    def sample(self, time, stop_vector, state, name=None):
        """sample stop signal."""
        del time, state  # unused by sample_fn
        stop_id = math_ops.cast(
            math_ops.argmax(stop_vector, axis=-1), dtypes.int32)
        return stop_id


class VectorGreedyEmbeddingHelper(seq2seq.GreedyEmbeddingHelper):
    """A helper for use during inference.

    Pass output vectors as inputs to following sequences.
    Stop signal is predicted by additional layer.
    """

    def __init__(self, start_inputs):
        """Initializer.

        Args:
            start_inputs: inputs for the initial state.
        """
        self._batch_size = start_inputs.get_shape().as_list()[0]
        self._start_inputs = start_inputs

    @property
    def batch_size(self):
        return self._batch_size

    def initialize(self, name=None):
        finished = array_ops.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, stop_vector, state, name=None):
        """sample stop signal."""
        del time, state  # unused by sample_fn
        stop_id = math_ops.cast(
            math_ops.argmax(stop_vector, axis=-1), dtypes.int32)
        return stop_id

    def next_inputs(self, time, outputs, state, stop_id, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time  # unused by next_inputs_fn
        finished = math_ops.equal(stop_id, 1)  # 1 is stop signal
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self._start_inputs,
            lambda: outputs)
        return (finished, next_inputs, state)


class SyntacticGreedyEmbeddingHelper(seq2seq.Helper):
    """A helper for use during inference.

    Perform the argmax over the grammatically correct output candidates.
    Outputs are passes through an embedding layer to get the next input.
    """

    def __init__(self, dsl_syntax, max_program_len, embedding, start_tokens, end_token):
        """Initializer.

        Args:
            dsl_syntax: Syntax checker for generating next possible tokens.
            max_program_len: maximum program length
            embedding: A callable that takes a vector tensor of `ids` (argmax ids),
                or the `params` argument for `embedding_lookup`. The returned tensor
                will be passed to the decoder input.
            start_tokens: `int32` vector shaped `[batch_size]`, the start tokens
            end_token: `int32` scalar, the token that marks end of decoding.

        Raises:
            ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
                scalar.
        """
        self.dsl_syntax = dsl_syntax
        self.max_program_len = max_program_len
        self.previous_tokens = []
        self.previous_probs = []
        self.previous_masks = []

        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
                lambda ids: embedding_ops.embedding_lookup(embedding, ids))

        self._start_tokens = ops.convert_to_tensor(
            start_tokens, dtype=dtypes.int32, name="start_tokens")
        self._end_token = ops.convert_to_tensor(
            end_token, dtype=dtypes.int32, name="end_token")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = array_ops.size(start_tokens)
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")
        self._start_inputs = self._embedding_fn(self._start_tokens)

    @property
    def batch_size(self):
        return self._batch_size

    def initialize(self, name=None):
        finished = array_ops.tile([False], [self._batch_size])

        def init_prev_tokens(finished):
            self.previous_tokens = []
            self.previous_probs = []
            self.previous_masks = []
            return finished
        new_finished = tf.py_func(init_prev_tokens, [finished], tf.bool)
        new_finished.set_shape(finished.get_shape())
        return (new_finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        """sample for SyntacticGreedyEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))

        # Mask outputs to reduce candidates to syntatically correct ones.
        def mask_output(outputs, end_token):
            if len(self.previous_tokens) == 0:  # when there is no previous token, skip masking.
                mask = np.zeros(outputs.shape, dtype=outputs.dtype)
                mask[:, self.dsl_syntax.token2int['DEF']] = 1
                return mask
            tokens = np.stack(self.previous_tokens, axis=1)
            masks = []
            for i in range(outputs.shape[0]):
                if tokens[i][-1] == end_token:
                    next_tokens = [end_token]
                else:
                    try:
                        p_str = self.dsl_syntax.intseq2str(tokens[i])
                        next_tokens_with_counts = self.dsl_syntax.get_next_candidates(p_str)
                        next_tokens = [t[0] for t in next_tokens_with_counts
                                       if t[1] <= self.max_program_len - len(tokens[i])]
                    except:
                        # TODO: this code rarely cause syntax error, which
                        # should not happen. We should fix this in the future.
                        next_tokens = [t for t in range(len(self.dsl_syntax.int2token))]
                    else:
                        next_tokens = [self.dsl_syntax.token2int[t] for t in next_tokens]
                mask = np.zeros([outputs.shape[1]], dtype=outputs.dtype)
                for t in next_tokens:
                    mask[t] = 1
                masks.append(mask)
            return np.stack(masks, axis=0)
        masks = tf.py_func(mask_output, [outputs, self._end_token], tf.float32)
        masks.set_shape(outputs.get_shape())
        masked_outputs = tf.exp(outputs) * masks
        sample_ids = math_ops.cast(
            math_ops.argmax(masked_outputs, axis=-1), dtypes.int32)

        def add_sample_ids(sample_ids):
            self.previous_tokens.append(sample_ids)
            return sample_ids
        new_sample_ids = tf.py_func(add_sample_ids, [sample_ids], tf.int32)
        new_sample_ids.set_shape(sample_ids.get_shape())
        return new_sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time, outputs  # unused by next_inputs_fn
        finished = math_ops.equal(sample_ids, self._end_token)
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self._start_inputs,
            lambda: self._embedding_fn(sample_ids))
        return (finished, next_inputs, state)


class SyntacticSampleEmbeddingHelper(SyntacticGreedyEmbeddingHelper):
    """A helper for use during inference.

    Uses sampling (from a distribution) instead of argmax and passes the
    result through an embedding layer to get the next input.
    """

    def __init__(self, dsl_syntax, max_program_len, embedding, start_tokens, end_token, seed=None):
        """Initializer.

        Args:
            dsl_syntax: Syntax checker for generating next possible tokens.
            max_program_len: maximum program length
            embedding: A callable that takes a vector tensor of `ids` (argmax ids),
                or the `params` argument for `embedding_lookup`. The returned tensor
                will be passed to the decoder input.
            start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
            end_token: `int32` scalar, the token that marks end of decoding.
            seed: The sampling seed.

        Raises:
            ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
                scalar.
        """
        super(SyntacticSampleEmbeddingHelper, self).__init__(
            dsl_syntax, max_program_len, embedding, start_tokens, end_token)
        self._seed = seed

    def sample(self, time, outputs, state, name=None):
        """sample for SyntacticGreedyEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))

        # Mask outputs to reduce candidates to syntatically correct ones.
        def mask_output(outputs, end_token):
            if len(self.previous_tokens) == 0:  # when there is no previous token, skip masking.
                mask = np.zeros(outputs.shape, dtype=outputs.dtype)
                mask[:, self.dsl_syntax.token2int['DEF']] = 1
                return mask
            tokens = np.stack(self.previous_tokens, axis=1)
            masks = []
            for i in range(outputs.shape[0]):
                if tokens[i][-1] == end_token:
                    next_tokens = [end_token]
                else:
                    try:
                        p_str = self.dsl_syntax.intseq2str(tokens[i])
                        next_tokens_with_counts = self.dsl_syntax.get_next_candidates(
                            '{}'.format(p_str))
                        next_tokens = [t[0] for t in next_tokens_with_counts
                                       if t[1] <= self.max_program_len - len(tokens[i])]
                    except:
                        # TODO: this code rarely cause syntax error, which
                        # should not happen. We should fix this in the future.
                        next_tokens = [t for t in range(len(self.dsl_syntax.int2token))]
                    else:
                        next_tokens = [self.dsl_syntax.token2int[t] for t in next_tokens]
                mask = np.zeros([outputs.shape[1]], dtype=outputs.dtype)
                for t in next_tokens:
                    mask[t] = 1
                masks.append(mask)
            return np.stack(masks, axis=0)
        masks = tf.py_func(mask_output, [outputs, self._end_token], tf.float32)
        masks.set_shape(outputs.get_shape())
        masked_outputs = tf.exp(outputs) * masks
        masked_probs = masked_outputs / \
            tf.reduce_sum(masked_outputs, axis=1, keep_dims=True)
        sample_id_sampler = categorical.Categorical(probs=masked_probs)
        sample_ids = sample_id_sampler.sample(seed=self._seed)

        def add_sample_ids(sample_ids, masked_probs, masks):
            self.previous_tokens.append(sample_ids)
            self.previous_probs.append(masked_probs)
            self.previous_masks.append(masks)

            return sample_ids
        new_sample_ids = tf.py_func(add_sample_ids, [sample_ids, masked_probs, masks], tf.int32)
        new_sample_ids.set_shape(sample_ids.get_shape())
        return new_sample_ids
