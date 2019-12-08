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
"""BERT model input pipelines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import tensorflow as tf

from data import bert_tokenization
from data.bert_classifier_data_lib import file_based_convert_examples_to_features


class BertClassifierDataset(object):
  def __init__(self, data_dir):
    self.data_dir = data_dir
    self.metadata = None

  def get_meta_data(self):
    if not self.metadata:
      with tf.io.gfile.GFile(self._get_metadata_path(), 'rb') as reader:
        self.meta_data = json.loads(reader.read().decode('utf-8'))
    return self.meta_data

  def get_dataset(self, set_type, batch_size, is_training=None):
    self.metadata = self.get_meta_data()
    if is_training is None:
      is_training = set_type == 'train'
    return create_classifier_dataset(
      self._get_tf_record_path(set_type),
      seq_length=self.metadata['max_seq_length'],
      batch_size=batch_size,
      is_training=is_training,
      drop_remainder=is_training)

  def gen_tf_records(self, processor, vocab_file, max_seq_length, do_lower_case=True):
    tokenizer = bert_tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

    meta_data = {
      "labels": processor.get_labels(),
      "num_labels": len(processor.get_labels()),
      "max_seq_length": max_seq_length,
    }
    for set_type in processor.get_set_types():
      tf_record_path = self._get_tf_record_path(set_type)
      input_data_examples = processor.get_examples(self.data_dir, set_type)
      file_based_convert_examples_to_features(input_data_examples,
                                              processor.get_labels(),
                                              max_seq_length, tokenizer,
                                              tf_record_path)
      meta_data['{}_data_size'.format(set_type)] = len(input_data_examples)
      if set_type == 'dev':
        meta_data['eval_data_size'.format(set_type)] = len(input_data_examples)

    with tf.io.gfile.GFile(self._get_metadata_path(), "w") as writer:
      writer.write(json.dumps(meta_data, indent=4) + "\n")

  def _get_tf_record_path(self, set_type):
    return os.path.join(self.data_dir, '{}.tf_record'.format(set_type))

  def _get_metadata_path(self):
    return os.path.join(self.data_dir, 'meta_data')


def create_classifier_dataset(file_path,
                              seq_length,
                              batch_size,
                              is_training=True,
                              drop_remainder=True):
  """Creates input dataset from (tf)records files for train/eval."""
  name_to_features = {
      'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'label_ids': tf.io.FixedLenFeature([], tf.int64),
      'is_real_example': tf.io.FixedLenFeature([], tf.int64),
  }
  input_fn = _file_based_input_fn_builder(file_path, name_to_features)
  dataset = input_fn()

  def _select_data_from_record(record):
    x = {
        'input_word_ids': record['input_ids'],
        'input_mask': record['input_mask'],
        'input_type_ids': record['segment_ids']
    }
    y = record['label_ids']
    return (x, y)

  dataset = dataset.map(_select_data_from_record)

  if is_training:
    dataset = dataset.shuffle(100)
    dataset = dataset.repeat()

  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  dataset = dataset.prefetch(1024)
  return dataset


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, tf.int32)
    example[name] = t

  return example


def _file_based_input_fn_builder(input_file, name_to_features):
  """Creates an `input_fn` closure to be passed for BERT custom training."""

  def input_fn():
    """Returns dataset for training/evaluation."""
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    d = d.map(lambda record: _decode_record(record, name_to_features))

    # When `input_file` is a path to a single file or a list
    # containing a single path, disable auto sharding so that
    # same input file is sent to all workers.
    if isinstance(input_file, str) or len(input_file) == 1:
      options = tf.data.Options()
      #options.experimental_distribute.auto_shard_policy = (
      #    tf.data.experimental.AutoShardPolicy.OFF)
      d = d.with_options(options)
    return d

  return input_fn
