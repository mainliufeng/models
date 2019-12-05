import tensorflow as tf

from utils import tf_utils


class EmbeddingLookup(tf.keras.layers.Layer):
  """Looks up words embeddings for id tensor."""

  def __init__(self,
               vocab_size,
               embedding_size=768,
               initializer="glorot_uniform",
               **kwargs):
    super(EmbeddingLookup, self).__init__(**kwargs)
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.initializer = initializer

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.embeddings = self.add_weight(
        "embeddings",
        shape=[self.vocab_size, self.embedding_size],
        initializer=self.initializer,
        dtype=self.dtype)
    super(EmbeddingLookup, self).build(unused_input_shapes)

  def call(self, inputs):
    """Implements call() for the layer."""
    input_shape = tf_utils.get_shape_list(inputs)
    flat_input = tf.reshape(inputs, [-1])
    output = tf.gather(self.embeddings, flat_input)
    output = tf.reshape(output, input_shape + [self.embedding_size])
    return output


class EmbeddingLookupFactorized(tf.keras.layers.Layer):
  """Looks up words embeddings for id tensor."""

  def __init__(self,
               vocab_size,
               hidden_size,
               embedding_size=768,
               initializer="glorot_uniform",
               **kwargs):
    super(EmbeddingLookupFactorized, self).__init__(**kwargs)
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.embedding_size = embedding_size
    self.initializer = initializer

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.embeddings = self.add_weight(
        "embeddings",
        shape=[self.vocab_size, self.embedding_size],
        initializer=self.initializer,
        dtype=self.dtype)
    self.project_variable = self.add_weight(
        "embeddings_2",
        shape=[self.embedding_size, self.hidden_size],
        initializer=self.initializer,
        dtype=self.dtype)
    super(EmbeddingLookupFactorized, self).build(unused_input_shapes)

  def call(self, inputs):
    """Implements call() for the layer."""
    input_shape = tf_utils.get_shape_list(inputs)
    flat_input = tf.reshape(inputs, [-1])
    output = tf.gather(self.embeddings, flat_input)
    output = tf.matmul(output, self.project_variable)
    output = tf.reshape(output, input_shape + [self.hidden_size])
    return output


class EmbeddingPostprocessor(tf.keras.layers.Layer):
  """Performs various post-processing on a word embedding tensor."""

  def __init__(self,
               use_type_embeddings=False,
               token_type_vocab_size=None,
               use_position_embeddings=True,
               max_position_embeddings=512,
               dropout_prob=0.0,
               initializer="glorot_uniform",
               **kwargs):
    super(EmbeddingPostprocessor, self).__init__(**kwargs)
    self.use_type_embeddings = use_type_embeddings
    self.token_type_vocab_size = token_type_vocab_size
    self.use_position_embeddings = use_position_embeddings
    self.max_position_embeddings = max_position_embeddings
    self.dropout_prob = dropout_prob
    self.initializer = initializer

    if self.use_type_embeddings and not self.token_type_vocab_size:
      raise ValueError("If `use_type_embeddings` is True, then "
                       "`token_type_vocab_size` must be specified.")

  def build(self, input_shapes):
    """Implements build() for the layer."""
    (word_embeddings_shape, _) = input_shapes
    width = word_embeddings_shape.as_list()[-1]
    self.type_embeddings = None
    if self.use_type_embeddings:
      self.type_embeddings = self.add_weight(
          "type_embeddings",
          shape=[self.token_type_vocab_size, width],
          initializer=self.initializer,
          dtype=self.dtype)

    self.position_embeddings = None
    if self.use_position_embeddings:
      self.position_embeddings = self.add_weight(
          "position_embeddings",
          shape=[self.max_position_embeddings, width],
          initializer=self.initializer,
          dtype=self.dtype)

    self.output_layer_norm = tf.keras.layers.LayerNormalization(
        name="layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)
    self.output_dropout = tf.keras.layers.Dropout(rate=self.dropout_prob,
                                                  dtype=tf.float32)
    super(EmbeddingPostprocessor, self).build(input_shapes)

  def __call__(self, word_embeddings, token_type_ids=None, **kwargs):
    inputs = tf_utils.pack_inputs([word_embeddings, token_type_ids])
    return super(EmbeddingPostprocessor, self).__call__(inputs, **kwargs)

  def call(self, inputs):
    """Implements call() for the layer."""
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    word_embeddings = unpacked_inputs[0]
    token_type_ids = unpacked_inputs[1]
    input_shape = tf_utils.get_shape_list(word_embeddings, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = word_embeddings
    if self.use_type_embeddings:
      flat_token_type_ids = tf.reshape(token_type_ids, [-1])
      token_type_embeddings = tf.gather(self.type_embeddings,
                                        flat_token_type_ids)
      token_type_embeddings = tf.reshape(token_type_embeddings,
                                         [batch_size, seq_length, width])
      output += token_type_embeddings

    if self.use_position_embeddings:
      position_embeddings = tf.expand_dims(
          tf.slice(self.position_embeddings, [0, 0], [seq_length, width]),
          axis=0)

      output += position_embeddings

    output = self.output_layer_norm(output)
    output = self.output_dropout(output)

    return output