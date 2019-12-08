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

import math
import os

from absl import app
from absl import flags
import tensorflow as tf

import bert.bert_config
from bert.bert_classifier import bert_classifier_model
from optimization.optimizers import adamw_polynomial_decay_warmup
from dataset.bert_classifier_dataset import BertClassifierDataset
from loss.losses import get_loss_fn
from flags import common_bert_flags, train_bert_flags
from utils import settings


common_bert_flags.define_common_bert_flags()
train_bert_flags.define_train_bert_flags()

FLAGS = flags.FLAGS


def main(_):
  settings.common_settings()

  # dataset and config
  dataset = BertClassifierDataset(FLAGS.input_data_dir)
  input_meta_data = dataset.get_meta_data()
  training_dataset = dataset.get_dataset('train', FLAGS.train_batch_size)
  evaluation_dataset = dataset.get_dataset('dev', FLAGS.train_batch_size)

  bert_config = bert.bert_config.BertConfig.from_json_file(FLAGS.bert_config_file)
  epochs = int(FLAGS.num_train_epochs)
  steps_per_epoch = int(input_meta_data['train_data_size'] / FLAGS.train_batch_size)
  num_train_steps = steps_per_epoch * epochs
  steps_per_eval_epoch = int(math.ceil(input_meta_data['eval_data_size'] / FLAGS.eval_batch_size))
  warmup_steps = int(epochs * input_meta_data['train_data_size'] * 0.1 / FLAGS.train_batch_size)

  # model
  model, bert_model = (
    bert_classifier_model(
      bert_config,
      tf.float32,
      input_meta_data['num_labels'],
      input_meta_data['max_seq_length'],
      share_parameter_across_layers=FLAGS.share_parameter_across_layers))

  # optimizer
  optimizer = adamw_polynomial_decay_warmup(
    num_train_steps, warmup_steps,
    learning_rate=FLAGS.learning_rate,
    fp16=FLAGS.fp16)

  # loss
  loss_fn = get_loss_fn(
    loss=None,
    num_train_steps=num_train_steps,
    num_classes=input_meta_data['num_labels'])

  # initialize bert core model
  if FLAGS.init_checkpoint:
    checkpoint = tf.train.Checkpoint(model=bert_model)
    checkpoint.restore(FLAGS.init_checkpoint).assert_nontrivial_match()

  # metrics
  metrics = [
    tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32),
  ]

  model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

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

  model.fit(
      x=training_dataset,
      validation_data=evaluation_dataset,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      validation_steps=steps_per_eval_epoch,
      callbacks=custom_callbacks)


if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('input_data_dir')
  flags.mark_flag_as_required('model_dir')
  app.run(main)
