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
"""Run masked LM/next sentence masked_lm pre-training for BERT in tf2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from flags import common_bert_flags
from dataset.bert_pretrain_dataset import create_pretrain_dataset
from bert.bert_config import BertConfig
from bert import bert_pretraining
from utils import bert_model_saving_utils
from optimization.optimizers import adamw_polynomial_decay_warmup


flags.DEFINE_string('input_files', None,
                    'File path to retrieve training data for pre-training.')
# Model training specific flags.
flags.DEFINE_integer(
    'max_seq_length', 128,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')
flags.DEFINE_integer('max_predictions_per_seq', 20,
                     'Maximum predictions per sequence_output.')
flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')
flags.DEFINE_integer('num_steps_per_epoch', 1000,
                     'Total number of training steps to run per epoch.')
flags.DEFINE_float('warmup_steps', 10000,
                   'Warmup steps for Adam weight decay optimizer.')

common_bert_flags.define_common_bert_flags()

FLAGS = flags.FLAGS


def main(_):
  # Users should always run this script under TF 2.x
  assert tf.version.VERSION.startswith('2.')

  bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)

  input_files = []
  for input_pattern in FLAGS.input_files.split(','):
    input_files.extend(tf.io.gfile.glob(input_pattern))

  train_dataset = create_pretrain_dataset(
      input_files,
      FLAGS.max_seq_length,
      FLAGS.max_predictions_per_seq,
      FLAGS.train_batch_size,
      is_training=True)

  pretrain_model, core_model = bert_pretraining.pretrain_model(
      bert_config, FLAGS.max_seq_length, FLAGS.max_predictions_per_seq)
  pretrain_model.optimizer = adamw_polynomial_decay_warmup(
    FLAGS.num_train_epochs * FLAGS.num_steps_per_epoch,
    FLAGS.warmup_steps,
    learning_rate=FLAGS.learning_rate)

  def bert_pretrain_loss_fn(unused_labels, losses, **unused_args):
    return tf.keras.backend.mean(losses)

  if FLAGS.init_checkpoint:
    logging.info('Checkpoint initialized from %s.', FLAGS.init_checkpoint)
    checkpoint = tf.train.Checkpoint(model=core_model)
    checkpoint.restore(FLAGS.init_checkpoint).assert_nontrivial_match()

  pretrain_model.compile(optimizer=pretrain_model.optimizer,
                         loss=bert_pretrain_loss_fn)

  checkpoint_path = os.path.join(FLAGS.model_dir, 'checkpoint')
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      checkpoint_path, save_weights_only=True)
  custom_callbacks = [checkpoint_callback]

  pretrain_model.fit(
    x=train_dataset,
    steps_per_epoch=FLAGS.num_steps_per_epoch,
    epochs=int(FLAGS.num_train_epochs),
    callbacks=custom_callbacks)

  bert_model_saving_utils.export_pretraining_checkpoint(
      checkpoint_dir=FLAGS.model_dir, model=core_model)


if __name__ == '__main__':
  app.run(main)
