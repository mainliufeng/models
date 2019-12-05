import tensorflow as tf

from modeling.attention import Attention
from modeling.dense import Dense2DProjection, Dense3D

from utils import tf_utils


class Transformer(tf.keras.layers.Layer):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
  """

  def __init__(self,
               num_hidden_layers=12,
               hidden_size=768,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_activation="gelu",
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer="glorot_uniform",
               backward_compatible=False,
               float_type=tf.float32,
               share_parameter_across_layers=False,
               **kwargs):
    super(Transformer, self).__init__(**kwargs)
    self.num_hidden_layers = num_hidden_layers
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf_utils.get_activation(
        intermediate_activation)
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer = initializer
    self.backward_compatible = backward_compatible
    self.float_type = float_type
    self.share_parameter_across_layers = share_parameter_across_layers

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.layers = []
    layer = None
    for i in range(self.num_hidden_layers):
      if self.share_parameter_across_layers:
        if layer is None:
          layer = TransformerBlock(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            intermediate_activation=self.intermediate_activation,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer=self.initializer,
            backward_compatible=self.backward_compatible,
            float_type=self.float_type,
            name="layer_shared")
      else:
        layer = TransformerBlock(
          hidden_size=self.hidden_size,
          num_attention_heads=self.num_attention_heads,
          intermediate_size=self.intermediate_size,
          intermediate_activation=self.intermediate_activation,
          hidden_dropout_prob=self.hidden_dropout_prob,
          attention_probs_dropout_prob=self.attention_probs_dropout_prob,
          initializer=self.initializer,
          backward_compatible=self.backward_compatible,
          float_type=self.float_type,
          name=("layer_%d" % i))
      self.layers.append(layer)
    super(Transformer, self).build(unused_input_shapes)

  def __call__(self, input_tensor, attention_mask=None, **kwargs):
    inputs = tf_utils.pack_inputs([input_tensor, attention_mask])
    return super(Transformer, self).__call__(inputs=inputs, **kwargs)

  def call(self, inputs, return_all_layers=False):
    """Implements call() for the layer.

    Args:
      inputs: packed inputs.
      return_all_layers: bool, whether to return outputs of all layers inside
        encoders.
    Returns:
      Output tensor of the last layer or a list of output tensors.
    """
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    input_tensor = unpacked_inputs[0]
    attention_mask = unpacked_inputs[1]
    output_tensor = input_tensor

    all_layer_outputs = []
    for layer in self.layers:
      output_tensor = layer(output_tensor, attention_mask)
      all_layer_outputs.append(output_tensor)

    if return_all_layers:
      return all_layer_outputs

    return all_layer_outputs[-1]


class TransformerBlock(tf.keras.layers.Layer):
  """Single transformer layer.

  It has two sub-layers. The first is a multi-head self-attention mechanism, and
  the second is a positionwise fully connected feed-forward network.
  """

  def __init__(self,
               hidden_size=768,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_activation="gelu",
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer="glorot_uniform",
               backward_compatible=False,
               float_type=tf.float32,
               **kwargs):
    super(TransformerBlock, self).__init__(**kwargs)
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf_utils.get_activation(
        intermediate_activation)
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer = initializer
    self.backward_compatible = backward_compatible
    self.float_type = float_type

    if self.hidden_size % self.num_attention_heads != 0:
      raise ValueError(
          "The hidden size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (self.hidden_size, self.num_attention_heads))
    self.attention_head_size = int(self.hidden_size / self.num_attention_heads)

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.attention_layer = Attention(
        num_attention_heads=self.num_attention_heads,
        size_per_head=self.attention_head_size,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        initializer=self.initializer,
        backward_compatible=self.backward_compatible,
        name="self_attention")
    self.attention_output_dense = Dense3D(
        num_attention_heads=self.num_attention_heads,
        size_per_head=int(self.hidden_size / self.num_attention_heads),
        kernel_initializer=self.initializer,
        output_projection=True,
        backward_compatible=self.backward_compatible,
        name="self_attention_output")
    self.attention_dropout = tf.keras.layers.Dropout(
        rate=self.hidden_dropout_prob)
    self.attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm", axis=-1, epsilon=1e-12,
            # We do layer norm in float32 for numeric stability.
            dtype=tf.float32))
    self.intermediate_dense = Dense2DProjection(
        output_size=self.intermediate_size,
        kernel_initializer=self.initializer,
        activation=self.intermediate_activation,
        # Uses float32 so that gelu activation is done in float32.
        fp32_activation=True,
        name="intermediate")
    self.output_dense = Dense2DProjection(
        output_size=self.hidden_size,
        kernel_initializer=self.initializer,
        name="output")
    self.output_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)
    self.output_layer_norm = tf.keras.layers.LayerNormalization(
        name="output_layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)
    super(TransformerBlock, self).build(unused_input_shapes)

  def common_layers(self):
    """Explicitly gets all layer objects inside a Transformer encoder block."""
    return [
        self.attention_layer, self.attention_output_dense,
        self.attention_dropout, self.attention_layer_norm,
        self.intermediate_dense, self.output_dense, self.output_dropout,
        self.output_layer_norm
    ]

  def __call__(self, input_tensor, attention_mask=None, **kwargs):
    inputs = tf_utils.pack_inputs([input_tensor, attention_mask])
    return super(TransformerBlock, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    (input_tensor, attention_mask) = tf_utils.unpack_inputs(inputs)
    attention_output = self.attention_layer(
        from_tensor=input_tensor,
        to_tensor=input_tensor,
        attention_mask=attention_mask)
    attention_output = self.attention_output_dense(attention_output)
    attention_output = self.attention_dropout(attention_output)
    # Use float32 in keras layer norm and the gelu activation in the
    # intermediate dense layer for numeric stability
    attention_output = self.attention_layer_norm(input_tensor +
                                                 attention_output)
    if self.float_type == tf.float16:
      attention_output = tf.cast(attention_output, tf.float16)
    intermediate_output = self.intermediate_dense(attention_output)
    if self.float_type == tf.float16:
      intermediate_output = tf.cast(intermediate_output, tf.float16)
    layer_output = self.output_dense(intermediate_output)
    layer_output = self.output_dropout(layer_output)
    # Use float32 in keras layer norm for numeric stability
    layer_output = self.output_layer_norm(layer_output + attention_output)
    if self.float_type == tf.float16:
      layer_output = tf.cast(layer_output, tf.float16)
    return layer_output

