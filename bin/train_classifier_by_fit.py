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
  epochs = int(FLAGS.num_train_epochs)
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

  # fp16
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

  # metrics
  def metric_fn():
    return [
      tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32),
    ]

  classifier_model.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric_fn()])

  # callbacks
  summary_dir = os.path.join(FLAGS.model_dir, 'summaries')
  summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
  checkpoint_path = os.path.join(FLAGS.model_dir, 'checkpoint')
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      checkpoint_path, save_weights_only=True)

  custom_callbacks = [
    summary_callback,
    checkpoint_callback,
  ]

  classifier_model.fit(
      x=training_dataset,
      validation_data=evaluation_dataset,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      validation_steps=steps_per_eval_epoch,
      callbacks=custom_callbacks)

  return classifier_model


if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('input_meta_data_path')
  flags.mark_flag_as_required('model_dir')
  app.run(main)
