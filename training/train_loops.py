import os

import tensorflow as tf
from absl import logging


def run_custom_train_loop(bert_model, model, optimizer, loss_fn,
                          training_dataset, evaluation_dataset,
                          epochs, steps_per_epoch, steps_per_eval_epoch,
                          init_checkpoint, model_dir,
                          metrics_fn=None):
  # initialize bert core model
  if init_checkpoint:
    checkpoint = tf.train.Checkpoint(model=bert_model)
    checkpoint.restore(init_checkpoint).assert_nontrivial_match()

  # initialize or load classifier model
  checkpoint = tf.train.Checkpoint(
    step=tf.Variable(1), optimizer=optimizer, model=model)
  manager = tf.train.CheckpointManager(checkpoint, model_dir, max_to_keep=3)
  checkpoint.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    logging.info('Checkpoint restored from %s.', manager.latest_checkpoint)
  else:
    logging.info('Checkpoint initializing from scratch.')

  # scalar
  summary_dir = os.path.join(model_dir, 'summaries')
  train_summary_writer = tf.summary.create_file_writer(summary_dir + '/train')
  test_summary_writer = tf.summary.create_file_writer(summary_dir + '/test')

  # metrics
  train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
  eval_metrics = metrics_fn() if metrics_fn else []
  eval_loss_metric = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)
  train_metrics = [
    metric.__class__.from_config(metric.get_config())
    for metric in eval_metrics
  ]

  @tf.function
  def train_for_steps(train_iter, steps):
    for _ in tf.range(steps):
      train_one_step(train_iter)

  def train_one_step(train_iter):
    x_batch_train, y_batch_train = next(train_iter)
    with tf.GradientTape() as tape:
      logits = model(x_batch_train)
      loss = loss_fn(y_batch_train, logits)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_loss_metric.update_state(loss)
    for metric in train_metrics:
      metric.update_state(y_batch_train, logits)

  @tf.function
  def eval_for_steps(eval_iter, steps):
    for _ in tf.range(steps):
      eval_one_step(eval_iter)

  def eval_one_step(eval_iter):
    x_batch_val, y_batch_val = next(eval_iter)
    val_logits = model(x_batch_val)
    loss = loss_fn(y_batch_val, val_logits)
    eval_loss_metric.update_state(loss)
    for metric in eval_metrics:
      metric.update_state(y_batch_val, val_logits)

  total_training_steps = steps_per_epoch * epochs

  train_iter = iter(training_dataset)

  steps_per_train = 100
  steps_per_eval = 1000
  current_step = int(checkpoint.step)
  while current_step < total_training_steps:
    steps = steps_to_run(current_step, steps_per_train, steps_per_eval)
    train_for_steps(train_iter, steps)
    current_step += steps
    checkpoint.step.assign(current_step)

    train_loss = train_loss_metric.result().numpy().astype(float)
    train_status = 'step: %s/%s, loss: %f' % (current_step, int(total_training_steps), train_loss)
    with train_summary_writer.as_default():
      tf.summary.scalar('loss', train_loss, step=current_step)
      tf.summary.scalar('learning_rate',
                        float(optimizer.learning_rate(current_step)),
                        step=current_step)
      for metric in train_metrics + model.metrics:
        metric_value = metric.result().numpy().astype(float)
        train_status += '  %s = %f' % (metric.name, metric_value)
        tf.summary.scalar(metric.name, metric_value, step=current_step)
    logging.info(train_status)

    if (current_step % steps_per_eval == 0
        or current_step >= total_training_steps):
      eval_iter = iter(evaluation_dataset)
      train_loss = train_loss_metric.result().numpy().astype(float)
      logging.info('train_loss: %f', train_loss)

      eval_for_steps(eval_iter, steps_per_eval_epoch)

      eval_loss = eval_loss_metric.result().numpy().astype(float)
      logging.info('eval loss: %f', eval_loss)
      with test_summary_writer.as_default():
        tf.summary.scalar('loss', eval_loss, step=current_step)
        for metric in eval_metrics + model.metrics:
          metric_value = metric.result().numpy().astype(float)
          logging.info('step: %s/%s, test %s = %f',
                       current_step, int(total_training_steps), metric.name, metric_value)
          tf.summary.scalar(
            metric.name, metric_value, step=current_step)
      eval_loss_metric.reset_states()
      for metric in eval_metrics + model.metrics:
        metric.reset_states()

      path = manager.save(checkpoint_number=current_step)
      logging.info("Checkpoint saved to %s", path)


def steps_to_run(current_step, steps_per_train, steps_per_eval):
  if steps_per_train <= 0:
    raise ValueError('steps_per_train should be positive integer.')
  if steps_per_train == 1:
    return steps_per_train
  remainder_in_eval = current_step % steps_per_eval
  remainder_in_train = current_step % steps_per_train
  if remainder_in_eval != 0:
    return min(steps_per_eval - remainder_in_eval,
               steps_per_train - remainder_in_train)
  else:
    return steps_per_train