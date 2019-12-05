import tensorflow as tf


def get_cross_entropy_loss(config):
  @tf.function
  def cross_entropy_loss(labels, logits):
    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=tf.stop_gradient(labels))
    loss = tf.reduce_mean(per_example_loss)
    return loss
  return cross_entropy_loss