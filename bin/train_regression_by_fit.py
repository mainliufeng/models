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

import os
import math

from absl import app
from absl import flags
import tensorflow as tf

import bert.bert_config
from bert.bert_regression import bert_regression_model
from optimization.optimizers import adamw_polynomial_decay_warmup
from dataset.bert_regression_dataset import BertRegressionDataset
from loss.losses import get_loss_fn
from flags import common_bert_flags, train_bert_flags
from utils import settings
from training.train_loops import run_custom_train_loop


common_bert_flags.define_common_bert_flags()
train_bert_flags.define_train_bert_flags()

flags.DEFINE_enum('loss', 'mse', ['mse', 'huber'], 'loss')
flags.DEFINE_float('delta', 1.0, 'delta of huber loss')

FLAGS = flags.FLAGS


def main(_):
  settings.common_settings()

  # dataset and config
  dataset = BertRegressionDataset(FLAGS.input_data_dir)
  input_meta_data = dataset.get_meta_data()
  training_dataset = dataset.get_dataset('train', FLAGS.train_batch_size)
  evaluation_dataset = dataset.get_dataset('dev', FLAGS.eval_batch_size)

  bert_config = bert.bert_config.BertConfig.from_json_file(FLAGS.bert_config_file)
  epochs = FLAGS.num_train_epochs
  steps_per_epoch = int(input_meta_data['train_data_size'] / FLAGS.train_batch_size)
  num_train_steps = steps_per_epoch * epochs
  steps_per_eval_epoch = int(math.ceil(input_meta_data['eval_data_size'] / FLAGS.eval_batch_size))
  warmup_steps = int(epochs * input_meta_data['train_data_size'] * 0.1 / FLAGS.train_batch_size)

  # model
  model, bert_model = (
    bert_regression_model(
      bert_config,
      tf.float32,
      input_meta_data['max_seq_length'],
      share_parameter_across_layers=FLAGS.share_parameter_across_layers,
      use_sigmoid=FLAGS.loss == 'mse'))

  # optimizer
  optimizer = adamw_polynomial_decay_warmup(
    num_train_steps, warmup_steps,
    learning_rate=FLAGS.learning_rate,
    fp16=FLAGS.fp16)

  # loss
  loss_fn = get_loss_fn(loss=FLAGS.loss, delta=FLAGS.delta)

  # initialize bert core model
  if FLAGS.init_checkpoint:
    checkpoint = tf.train.Checkpoint(model=bert_model)
    checkpoint.restore(FLAGS.init_checkpoint).assert_nontrivial_match()

  model.compile(optimizer=optimizer, loss=loss_fn)

  # callbacks
  summary_dir = os.path.join(FLAGS.model_dir, 'summaries')
  checkpoint_path = os.path.join(FLAGS.model_dir, 'checkpoint')
  best_checkpoint_path = os.path.join(FLAGS.model_dir, 'best_checkpoint')

  custom_callbacks = [
    tf.keras.callbacks.TensorBoard(summary_dir),
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True),
    tf.keras.callbacks.ModelCheckpoint(
      best_checkpoint_path, save_weights_only=True, save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
  ]

  model.fit(
      x=training_dataset,
      validation_data=evaluation_dataset,
      steps_per_epoch=steps_per_epoch,
      epochs=int(epochs),
      validation_steps=steps_per_eval_epoch,
      callbacks=custom_callbacks)


if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('input_data_dir')
  flags.mark_flag_as_required('model_dir')
  app.run(main)
