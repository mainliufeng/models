import tensorflow as tf
import tensorflow_addons as tfa
from functools import partial


EPSILON = 1e-30


def get_loss_fn(**config):
  if config.get('loss') == 'entropy':
    return get_cross_entropy_loss(config)
  elif config.get('loss') == 'focal':
    return get_focal_loss_multi_v1(config)
  else:
    return get_classification_loss(config)


def get_classification_loss(config):
  @tf.function
  def classification_loss(labels, logits):
    labels = tf.squeeze(labels)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(
        tf.cast(labels, dtype=tf.int32), depth=config.get('num_classes'), dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(
        tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return loss
  return classification_loss


def get_cross_entropy_loss(config):
  @tf.function
  def cross_entropy_loss(labels, logits):
    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=tf.stop_gradient(labels))
    loss = tf.reduce_mean(per_example_loss)
    return loss
  return cross_entropy_loss


def get_focal_loss_multi_v1(config):
  @tf.function
  def focal_loss_multi_v1(labels, logits):
    gamma = config.get("gamma", 2.0)
    labels = tf.cast(tf.expand_dims(labels, -1), tf.int32)
    predictions = tf.exp(tf.nn.log_softmax(logits, axis=-1))
    batch_idxs = tf.range(0, tf.shape(labels)[0])
    batch_idxs = tf.expand_dims(batch_idxs, 1)
    idxs = tf.concat([batch_idxs, labels], 1)
    y_true_pred = tf.gather_nd(predictions, idxs)
    labels = tf.cast(tf.squeeze(labels, axis=-1), tf.float32)
    losses =  tf.math.log(y_true_pred+EPSILON) * tf.pow(1-y_true_pred, gamma)
    #return -losses, y_true_pred
    return -losses
  return focal_loss_multi_v1

