import tensorflow as tf


EPSILON = 1e-30

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
    losses = tf.math.log(y_true_pred+EPSILON) * tf.pow(1-y_true_pred, gamma)
    #return -losses, y_true_pred
    return -losses
  return focal_loss_multi_v1