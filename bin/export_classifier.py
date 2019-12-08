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

import json

from absl import app
from absl import flags
import tensorflow as tf

import bert.bert_config
from bert.bert_classifier import bert_classifier_model
from flags import common_bert_flags as common_flags
from utils import bert_model_saving_utils as model_saving_utils

flags.DEFINE_string(
    'input_meta_data_path', None,
    'Path to file that contains meta data about input '
    'to be used for training and evaluation.')

common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS


def main(_):
  # Users should always run this script under TF 2.x
  assert tf.version.VERSION.startswith('2.')

  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

  bert_config = bert.bert_config.BertConfig.from_json_file(FLAGS.bert_config_file)

  classifier_model = bert_classifier_model(
      bert_config, tf.float32, input_meta_data['num_labels'],
      input_meta_data['max_seq_length'],
      share_parameter_across_layers=FLAGS.share_parameter_across_layers)[0]

  model_saving_utils.export_bert_model(
      FLAGS.model_export_path,
      model=classifier_model,
      checkpoint_dir=FLAGS.model_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('input_meta_data_path')
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('model_export_path')
  app.run(main)
