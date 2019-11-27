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
"""BERT classification finetuning runner in tf2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import math
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from bert import bert_modeling as modeling
from bert.bert_classifier import bert_classifier_model
from optimization import bert_optimization as optimization
from flags import common_bert_flags as common_flags
from data import bert_classifier_dataset

flags.DEFINE_string('train_data_path', None,
                    'Path to training data for BERT classifier.')
flags.DEFINE_string('eval_data_path', None,
                    'Path to evaluation data for BERT classifier.')
# Model training specific flags.
flags.DEFINE_string(
    'input_meta_data_path', None,
    'Path to file that contains meta data about input '
    'to be used for training and evaluation.')
flags.DEFINE_integer('train_batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('eval_batch_size', 32, 'Batch size for evaluation.')
flags.DEFINE_boolean('fp16', False, 'fp16')

common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS


def main(_):
  # Users should always run this script under TF 2.x
  assert tf.version.VERSION.startswith('2.')

  gpus = tf.config.experimental.list_physical_devices('GPU')
  assert len(gpus) > 0
  tf.config.experimental.set_memory_growth(gpus[0], True)

  # args
  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))
  train_data_size = input_meta_data['train_data_size']
  eval_data_size = input_meta_data['eval_data_size']
  max_seq_length = input_meta_data['max_seq_length']
  num_classes = input_meta_data['num_labels']

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  epochs = FLAGS.num_train_epochs
  steps_per_epoch = int(train_data_size / FLAGS.train_batch_size)
  steps_per_eval_epoch = int(math.ceil(eval_data_size / FLAGS.eval_batch_size))
  warmup_steps = int(epochs * train_data_size * 0.1 / FLAGS.train_batch_size)

  # dataset
  training_dataset = bert_classifier_dataset.create_classifier_dataset(
    FLAGS.train_data_path,
    seq_length=max_seq_length,
    batch_size=FLAGS.train_batch_size)
  train_iter = iter(training_dataset)
  evaluation_dataset = bert_classifier_dataset.create_classifier_dataset(
    FLAGS.eval_data_path,
    seq_length=max_seq_length,
    batch_size=FLAGS.eval_batch_size,
    is_training=False,
    drop_remainder=False)

  # model
  classifier_model, bert_core_model = (
    bert_classifier_model(
      bert_config,
      tf.float32,
      num_classes,
      max_seq_length,
      share_parameter_across_layers=FLAGS.share_parameter_across_layers))
  optimizer = optimization.create_optimizer(
    FLAGS.learning_rate, steps_per_epoch * epochs, warmup_steps)
  if FLAGS.fp16:
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
      optimizer)

  @tf.function
  def classification_loss(labels, logits):
    """Classification loss."""
    labels = tf.squeeze(labels)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(
        tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(
        tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return loss

  # initialize bert core model
  if FLAGS.init_checkpoint:
    checkpoint = tf.train.Checkpoint(model=bert_core_model)
    checkpoint.restore(FLAGS.init_checkpoint).assert_nontrivial_match()

  # initialize or load classifier model
  step = tf.Variable(1, name="step", dtype=tf.int64)
  checkpoint_path = os.path.join(FLAGS.model_dir, 'checkpoint')
  checkpoint = tf.train.Checkpoint(
    step=step, optimizer=optimizer, model=classifier_model)
  manager = tf.train.CheckpointManager(checkpoint, FLAGS.model_dir, max_to_keep=3)
  checkpoint.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    logging.info('Checkpoint restored from %s.', manager.latest_checkpoint)
  else:
    logging.info('Checkpoint initializing from scratch.')

  # scalar
  summary_dir = os.path.join(FLAGS.model_dir, 'summaries')
  train_summary_writer = tf.summary.create_file_writer(summary_dir + '/train')
  test_summary_writer = tf.summary.create_file_writer(summary_dir + '/test')

  # metrics
  train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
  eval_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
  train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
  eval_loss_metric = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)

  @tf.function
  def train_for_steps(train_iter, steps):
    for _ in tf.range(steps):
      train_one_step(train_iter)

  def train_one_step(train_iter):
    x_batch_train, y_batch_train = next(train_iter)
    with tf.GradientTape() as tape:
      logits = classifier_model(x_batch_train)
      loss = classification_loss(y_batch_train, logits)
    grads = tape.gradient(loss, classifier_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, classifier_model.trainable_weights))
    train_loss_metric.update_state(loss)
    train_acc_metric.update_state(y_batch_train, logits)

  @tf.function
  def eval_for_steps(eval_iter, steps):
    for _ in tf.range(steps):
      eval_one_step(eval_iter)

  def eval_one_step(eval_iter):
    x_batch_val, y_batch_val = next(eval_iter)
    val_logits = classifier_model(x_batch_val)
    loss = classification_loss(y_batch_val, val_logits)
    eval_loss_metric.update_state(loss)
    eval_acc_metric.update_state(y_batch_val, val_logits)

  total_training_steps = steps_per_epoch * epochs

  steps_per_train = 10
  steps_per_eval = 1000
  current_step = int(step)
  while current_step < total_training_steps:
    steps = steps_to_run(current_step, steps_per_train, steps_per_eval)
    train_for_steps(train_iter, steps)
    current_step += steps
    step.assign(current_step)

    train_loss = train_loss_metric.result().numpy().astype(float)
    train_acc = train_acc_metric.result().numpy().astype(float)
    logging.info('step: %s/%s, loss: %f',
                 int(step), int(total_training_steps), train_loss)
    with train_summary_writer.as_default():
      tf.summary.scalar('loss', train_loss, step=step)
      tf.summary.scalar('accuracy', train_acc, step=step)

    if (current_step % steps_per_eval == 0
        or current_step >= total_training_steps):
      eval_iter = iter(evaluation_dataset)
      train_loss = train_loss_metric.result().numpy().astype(float)
      train_acc = train_acc_metric.result().numpy().astype(float)
      logging.info('train_loss: %f, train accuracy: %f', train_loss, train_acc)

      eval_for_steps(eval_iter, steps_per_eval_epoch)

      eval_loss = eval_loss_metric.result().numpy().astype(float)
      eval_acc = eval_acc_metric.result().numpy().astype(float)
      logging.info('eval loss: %f, eval accuracy: %f', eval_loss, eval_acc)
      with test_summary_writer.as_default():
        tf.summary.scalar('loss', eval_loss, step=step)
        tf.summary.scalar('accuracy', eval_acc, step=step)
      eval_loss_metric.reset_states()
      eval_acc_metric.reset_states()

      path = manager.save(checkpoint_number=step)
      logging.info("Checkpoint saved to %s", path)

  train_acc_metric.reset_states()
  eval_acc_metric.reset_states()


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


if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('input_meta_data_path')
  flags.mark_flag_as_required('model_dir')
  app.run(main)
