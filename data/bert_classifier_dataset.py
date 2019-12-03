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

import tensorflow as tf


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


def _create_classifier_dataset(file_path,
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


def get_classifier_dataset(input_meta_data_path,
                           train_data_path, train_batch_size,
                           eval_data_path, eval_batch_size):
  with tf.io.gfile.GFile(input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))
  train_data_size = input_meta_data['train_data_size']
  eval_data_size = input_meta_data['eval_data_size']
  max_seq_length = input_meta_data['max_seq_length']
  num_classes = input_meta_data['num_labels']

  training_dataset = _create_classifier_dataset(
    train_data_path,
    seq_length=max_seq_length,
    batch_size=train_batch_size)
  evaluation_dataset = _create_classifier_dataset(
    eval_data_path,
    seq_length=max_seq_length,
    batch_size=eval_batch_size,
    is_training=False,
    drop_remainder=False)
  return input_meta_data, training_dataset, evaluation_dataset

