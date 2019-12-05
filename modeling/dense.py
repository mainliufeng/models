import tensorflow as tf


class Dense2DProjection(tf.keras.layers.Layer):
  """A 2D projection layer with tf.einsum implementation."""

  def __init__(self,
               output_size,
               kernel_initializer=None,
               bias_initializer="zeros",
               activation=None,
               fp32_activation=False,
               **kwargs):
    super(Dense2DProjection, self).__init__(**kwargs)
    self.output_size = output_size
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.activation = activation
    self.fp32_activation = fp32_activation

  def build(self, input_shape):
    """Implements build() for the layer."""
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError("Unable to build `Dense2DProjection` layer with "
                      "non-floating point (and non-complex) "
                      "dtype %s" % (dtype,))
    input_shape = tf.TensorShape(input_shape)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError("The last dimension of the inputs to "
                       "`Dense2DProjection` should be defined. "
                       "Found `None`.")
    last_dim = tf.compat.dimension_value(input_shape[-1])
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: last_dim})
    self.kernel = self.add_weight(
        "kernel",
        shape=[last_dim, self.output_size],
        initializer=self.kernel_initializer,
        dtype=self.dtype,
        trainable=True)
    self.bias = self.add_weight(
        "bias",
        shape=[self.output_size],
        initializer=self.bias_initializer,
        dtype=self.dtype,
        trainable=True)
    super(Dense2DProjection, self).build(input_shape)

  def call(self, inputs):
    """Implements call() for Dense2DProjection.

    Args:
      inputs: float Tensor of shape [batch, from_seq_length,
        num_attention_heads, size_per_head].

    Returns:
      A 3D Tensor.
    """
    ret = tf.einsum("abc,cd->abd", inputs, self.kernel)
    ret += self.bias
    if self.activation is not None:
      if self.dtype == tf.float16 and self.fp32_activation:
        ret = tf.cast(ret, tf.float32)
      return self.activation(ret)
    return ret


class Dense3D(tf.keras.layers.Layer):
  """A Dense Layer using 3D kernel with tf.einsum implementation.

  Attributes:
    num_attention_heads: An integer, number of attention heads for each
      multihead attention layer.
    size_per_head: An integer, hidden size per attention head.
    hidden_size: An integer, dimension of the hidden layer.
    kernel_initializer: An initializer for the kernel weight.
    bias_initializer: An initializer for the bias.
    activation: An activation function to use. If nothing is specified, no
      activation is applied.
    use_bias: A bool, whether the layer uses a bias.
    output_projection: A bool, whether the Dense3D layer is used for output
      linear projection.
    backward_compatible: A bool, whether the variables shape are compatible
      with checkpoints converted from TF 1.x.
  """

  def __init__(self,
               num_attention_heads=12,
               size_per_head=72,
               kernel_initializer=None,
               bias_initializer="zeros",
               activation=None,
               use_bias=True,
               output_projection=False,
               backward_compatible=False,
               **kwargs):
    """Inits Dense3D."""
    super(Dense3D, self).__init__(**kwargs)
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.hidden_size = num_attention_heads * size_per_head
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.activation = activation
    self.use_bias = use_bias
    self.output_projection = output_projection
    self.backward_compatible = backward_compatible

  @property
  def compatible_kernel_shape(self):
    if self.output_projection:
      return [self.hidden_size, self.hidden_size]
    return [self.last_dim, self.hidden_size]

  @property
  def compatible_bias_shape(self):
    return [self.hidden_size]

  @property
  def kernel_shape(self):
    if self.output_projection:
      return [self.num_attention_heads, self.size_per_head, self.hidden_size]
    return [self.last_dim, self.num_attention_heads, self.size_per_head]

  @property
  def bias_shape(self):
    if self.output_projection:
      return [self.hidden_size]
    return [self.num_attention_heads, self.size_per_head]

  def build(self, input_shape):
    """Implements build() for the layer."""
    dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError("Unable to build `Dense3D` layer with non-floating "
                      "point (and non-complex) dtype %s" % (dtype,))
    input_shape = tf.TensorShape(input_shape)
    if tf.compat.dimension_value(input_shape[-1]) is None:
      raise ValueError("The last dimension of the inputs to `Dense3D` "
                       "should be defined. Found `None`.")
    self.last_dim = tf.compat.dimension_value(input_shape[-1])
    self.input_spec = tf.keras.layers.InputSpec(
        min_ndim=3, axes={-1: self.last_dim})
    # Determines variable shapes.
    if self.backward_compatible:
      kernel_shape = self.compatible_kernel_shape
      bias_shape = self.compatible_bias_shape
    else:
      kernel_shape = self.kernel_shape
      bias_shape = self.bias_shape

    self.kernel = self.add_weight(
        "kernel",
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          "bias",
          shape=bias_shape,
          initializer=self.bias_initializer,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    super(Dense3D, self).build(input_shape)

  def call(self, inputs):
    """Implements ``call()`` for Dense3D.

    Args:
      inputs: A float tensor of shape [batch_size, sequence_length, hidden_size]
        when output_projection is False, otherwise a float tensor of shape
        [batch_size, sequence_length, num_heads, dim_per_head].

    Returns:
      The projected tensor with shape [batch_size, sequence_length, num_heads,
        dim_per_head] when output_projection is False, otherwise [batch_size,
        sequence_length, hidden_size].
    """
    if self.backward_compatible:
      kernel = tf.keras.backend.reshape(self.kernel, self.kernel_shape)
      bias = (tf.keras.backend.reshape(self.bias, self.bias_shape)
              if self.use_bias else None)
    else:
      kernel = self.kernel
      bias = self.bias

    if self.output_projection:
      ret = tf.einsum("abcd,cde->abe", inputs, kernel)
    else:
      ret = tf.einsum("abc,cde->abde", inputs, kernel)
    if self.use_bias:
      ret += bias
    if self.activation is not None:
      return self.activation(ret)
    return ret