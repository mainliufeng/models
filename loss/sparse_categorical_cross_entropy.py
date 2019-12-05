import tensorflow as tf


def get_sparse_categorical_cross_entropy_loss(config):
  @tf.function
  def sparse_categorical_cross_entropy_loss(labels, logits):
    labels = tf.squeeze(labels)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(
        tf.cast(labels, dtype=tf.int32), depth=config.get('num_classes'), dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(
        tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return loss
  return sparse_categorical_cross_entropy_loss