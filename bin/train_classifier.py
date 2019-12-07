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

import bert.bert_config
from bert import bert_model as modeling
from bert.bert_classifier import bert_classifier_model
from optimization.adamw import AdamWeightDecay
from optimization.warmup import WarmUp
from flags import common_bert_flags as common_flags
from dataset.bert_classifier_dataset import get_classifier_dataset as get_bert_classifier_dataset
from loss.losses import get_loss_fn

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
flags.DEFINE_string('loss', None, 'loss')

common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS


def main(_):
  # Users should always run this script under TF 2.x
  assert tf.version.VERSION.startswith('2.')

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

  # dataset and config
  (input_meta_data,
   training_dataset,
   evaluation_dataset) = get_bert_classifier_dataset(
    FLAGS.input_meta_data_path,
    FLAGS.train_data_path, FLAGS.train_batch_size,
    FLAGS.eval_data_path, FLAGS.eval_batch_size)

  bert_config = bert.bert_config.BertConfig.from_json_file(FLAGS.bert_config_file)
  epochs = FLAGS.num_train_epochs
  steps_per_epoch = int(input_meta_data['train_data_size'] / FLAGS.train_batch_size)
  num_train_steps = steps_per_epoch * epochs
  steps_per_eval_epoch = int(math.ceil(input_meta_data['eval_data_size'] / FLAGS.eval_batch_size))
  warmup_steps = int(epochs * input_meta_data['train_data_size'] * 0.1 / FLAGS.train_batch_size)

  train_iter = iter(training_dataset)

  # model
  classifier_model, bert_core_model = (
    bert_classifier_model(
      bert_config,
      tf.float32,
      input_meta_data['num_labels'],
      input_meta_data['max_seq_length'],
      share_parameter_across_layers=FLAGS.share_parameter_across_layers))

  # lr
  learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=FLAGS.learning_rate,
    decay_steps=num_train_steps,
    end_learning_rate=0.0,
  )

  learning_rate_fn = WarmUp(
    initial_learning_rate=FLAGS.learning_rate,
    decay_schedule_fn=learning_rate_fn,
    warmup_steps=warmup_steps,
  )

  # optimizer
  optimizer = AdamWeightDecay(
      learning_rate=learning_rate_fn,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=['layer_norm', 'bias'])

  if FLAGS.fp16:
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
      optimizer)

  # loss
  loss_fn = get_loss_fn(
    loss=None,
    num_train_steps=num_train_steps,
    num_classes=input_meta_data['num_labels'])

  # initialize bert core model
  if FLAGS.init_checkpoint:
    checkpoint = tf.train.Checkpoint(model=bert_core_model)
    checkpoint.restore(FLAGS.init_checkpoint).assert_nontrivial_match()

  # initialize or load classifier model
  checkpoint_path = os.path.join(FLAGS.model_dir, 'checkpoint')
  checkpoint = tf.train.Checkpoint(
    step=tf.Variable(1), optimizer=optimizer, model=classifier_model)
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
      loss = loss_fn(y_batch_train, logits)
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
    loss = loss_fn(y_batch_val, val_logits)
    eval_loss_metric.update_state(loss)
    eval_acc_metric.update_state(y_batch_val, val_logits)

  total_training_steps = steps_per_epoch * epochs

  steps_per_train = 100
  steps_per_eval = 1000
  current_step = int(checkpoint.step)
  while current_step < total_training_steps:
    steps = steps_to_run(current_step, steps_per_train, steps_per_eval)
    train_for_steps(train_iter, steps)
    current_step += steps
    checkpoint.step.assign(current_step)

    train_loss = train_loss_metric.result().numpy().astype(float)
    train_acc = train_acc_metric.result().numpy().astype(float)
    logging.info('step: %s/%s, loss: %f',
                 current_step, int(total_training_steps), train_loss)
    with train_summary_writer.as_default():
      tf.summary.scalar('loss', train_loss, step=current_step)
      tf.summary.scalar('accuracy', train_acc, step=current_step)
      tf.summary.scalar('learning_rate',
                        float(learning_rate_fn(current_step)),
                        step=current_step)

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
        tf.summary.scalar('loss', eval_loss, step=current_step)
        tf.summary.scalar('accuracy', eval_acc, step=current_step)
      eval_loss_metric.reset_states()
      eval_acc_metric.reset_states()

      path = manager.save(checkpoint_number=current_step)
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
