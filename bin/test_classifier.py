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

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import numpy as np

import bert.bert_config
from bert.bert_classifier import bert_classifier_model
from flags import common_bert_flags, test_bert_flags
from dataset.bert_classifier_dataset import create_classifier_dataset, BertClassifierDataset
from utils import settings


common_bert_flags.define_common_bert_flags()
test_bert_flags.define_test_bert_flags()

FLAGS = flags.FLAGS


def main(_):
  settings.common_settings()

  # dataset and config
  classifier_dataset = BertClassifierDataset(FLAGS.input_data_dir)
  dataset = classifier_dataset.get_dataset('dev', FLAGS.batch_size)
  meta_data = classifier_dataset.get_meta_data()

  bert_config = bert.bert_config.BertConfig.from_json_file(FLAGS.bert_config_file)

  # model
  model, bert_model = (
    bert_classifier_model(
      bert_config,
      tf.float32,
      meta_data['num_labels'],
      FLAGS.seq_length,
      share_parameter_across_layers=FLAGS.share_parameter_across_layers))

  # initialize or load classifier model
  checkpoint = tf.train.Checkpoint(model=model)
  manager = tf.train.CheckpointManager(checkpoint, FLAGS.model_dir, max_to_keep=3)
  checkpoint.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    logging.info('Checkpoint restored from %s.', manager.latest_checkpoint)
  else:
    exit(1)

  logging.info('predict start')
  prediction = model.predict(dataset)
  prediction = np.argmax(prediction, axis=-1)

  np.savetxt("test_results.csv", prediction, delimiter=",")


if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('input_data_dir')
  flags.mark_flag_as_required('model_dir')
  app.run(main)
