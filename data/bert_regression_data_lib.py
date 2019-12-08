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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os

from absl import logging
import tensorflow as tf

from data import bert_tokenization as tokenization


class InputExample(object):

  def __init__(self, guid, text_a, text_b=None, val=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      val: (Optional) float. The value of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.val = val


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               val,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.val = val
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_set_types(self):
    raise NotImplementedError()

  def get_set_path(self, set_type):
    raise NotImplementedError()

  def get_examples(self, input_file, set_type):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.io.gfile.GFile(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class StsProcessor(DataProcessor):
  """Processor for the STS-B data set."""

  def get_set_types(self):
    return ['train', 'dev', 'test']

  def get_set_path(self, data_dir, set_type):
    return os.path.join(data_dir, '{}.tsv'.format(set_type))

  def get_examples(self, data_dir, set_type):
    set_path = self.get_set_path(data_dir, set_type)
    if not os.path.exists(set_path) or not os.path.isfile(set_path):
      return
    return self._create_examples(self._read_tsv(set_path), set_type)

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[-3])
      text_b = tokenization.convert_to_unicode(line[-2])
      val = float(line[-1])
      examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, val=val))
    return examples


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  if ex_index < 5:
    logging.info("*** Example ***")
    logging.info("guid: %s", (example.guid))
    logging.info("tokens: %s",
                 " ".join([tokenization.printable_text(x) for x in tokens]))
    logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
    logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
    logging.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
    logging.info("vals: %s", example.val)

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      val=example.val,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(examples, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logging.info("Writing example %d of %d", ex_index, len(examples))

    feature = convert_single_example(ex_index, example, max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["vals"] = create_float_feature([feature.val])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()
