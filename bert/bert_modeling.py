# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

from bert.bert_config import BertConfig
from modeling.embedding import EmbeddingLookup, EmbeddingLookupFactorized, EmbeddingPostprocessor
from modeling.transformer import Transformer
from utils import tf_utils


def get_bert_model(input_word_ids,
                   input_mask,
                   input_type_ids,
                   config=None,
                   name=None,
                   float_type=tf.float32,
                   share_parameter_across_layers=False):
  """Wraps the core BERT model as a keras.Model."""
  bert_model_layer = BertModel(
    config=config, float_type=float_type,
    share_parameter_across_layers=share_parameter_across_layers, name=name)
  pooled_output, sequence_output = bert_model_layer(input_word_ids, input_mask,
                                                    input_type_ids)
  bert_model = tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=[pooled_output, sequence_output])
  return bert_model


class BertModel(tf.keras.layers.Layer):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_word_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  input_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  pooled_output, sequence_output = modeling.BertModel(config=config)(
    input_word_ids=input_word_ids,
    input_mask=input_mask,
    input_type_ids=input_type_ids)
  ...
  ```
  """

  def __init__(self, config, float_type=tf.float32,
               share_parameter_across_layers=False, **kwargs):
    super(BertModel, self).__init__(**kwargs)
    self.config = (
        BertConfig.from_dict(config)
        if isinstance(config, dict) else copy.deepcopy(config))
    self.float_type = float_type
    self.share_parameter_across_layers = share_parameter_across_layers

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    if self.config.embedding_size:
      self.embedding_lookup = EmbeddingLookupFactorized(
          vocab_size=self.config.vocab_size,
          hidden_size=self.config.hidden_size,
          embedding_size=self.config.embedding_size,
          dtype=tf.float32,
          name="word_embeddings")
    else:
      self.embedding_lookup = EmbeddingLookup(
          vocab_size=self.config.vocab_size,
          embedding_size=self.config.hidden_size,
          dtype=tf.float32,
          name="word_embeddings")
    self.embedding_postprocessor = EmbeddingPostprocessor(
        use_type_embeddings=True,
        token_type_vocab_size=self.config.type_vocab_size,
        use_position_embeddings=True,
        max_position_embeddings=self.config.max_position_embeddings,
        dropout_prob=self.config.hidden_dropout_prob,
        dtype=tf.float32,
        name="embedding_postprocessor")
    self.encoder = Transformer(
        num_hidden_layers=self.config.num_hidden_layers,
        hidden_size=self.config.hidden_size,
        num_attention_heads=self.config.num_attention_heads,
        intermediate_size=self.config.intermediate_size,
        intermediate_activation=self.config.hidden_act,
        hidden_dropout_prob=self.config.hidden_dropout_prob,
        attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
        backward_compatible=self.config.backward_compatible,
        float_type=self.float_type,
        share_parameter_across_layers=self.share_parameter_across_layers,
        name="encoder")
    self.pooler_transform = tf.keras.layers.Dense(
        units=self.config.hidden_size,
        activation="tanh",
        kernel_initializer="glorot_uniform",
        name="pooler_transform")
    super(BertModel, self).build(unused_input_shapes)

  def __call__(self,
               input_word_ids,
               input_mask=None,
               input_type_ids=None,
               **kwargs):
    inputs = tf_utils.pack_inputs([input_word_ids, input_mask, input_type_ids])
    return super(BertModel, self).__call__(inputs, **kwargs)

  def call(self, inputs, mode="bert"):
    """Implements call() for the layer.

    Args:
      inputs: packed input tensors.
      mode: string, `bert` or `encoder`.
    Returns:
      Output tensor of the last layer for BERT training (mode=`bert`) which
      is a float Tensor of shape [batch_size, seq_length, hidden_size] or
      a list of output tensors for encoder usage (mode=`encoder`).
    """
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    input_word_ids = unpacked_inputs[0]
    input_mask = unpacked_inputs[1]
    input_type_ids = unpacked_inputs[2]

    word_embeddings = self.embedding_lookup(input_word_ids)
    embedding_tensor = self.embedding_postprocessor(
        word_embeddings=word_embeddings, token_type_ids=input_type_ids)
    if self.float_type == tf.float16:
      embedding_tensor = tf.cast(embedding_tensor, tf.float16)
    attention_mask = None
    if input_mask is not None:
      attention_mask = create_attention_mask_from_input_mask(
          input_word_ids, input_mask)

    if mode == "encoder":
      return self.encoder(
          embedding_tensor, attention_mask, return_all_layers=True)

    sequence_output = self.encoder(embedding_tensor, attention_mask)
    first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
    pooled_output = self.pooler_transform(first_token_tensor)

    return (pooled_output, sequence_output)

  def get_config(self):
    config = {"config": self.config.to_dict()}
    base_config = super(BertModel, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = tf_utils.get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = tf_utils.get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]),
      dtype=from_tensor.dtype)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=from_tensor.dtype)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask
