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
import json

from absl import app
from absl import logging
from absl import flags
import tensorflow as tf

from data import bert_tokenization as tokenization


FLAGS = flags.FLAGS

# BERT classification specific flags.
flags.DEFINE_string(
    "input_data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

# Shared flags across BERT fine-tuning tasks.
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "train_data_output_path", None,
    "The path in which generated training input data will be written as tf"
    " records."
)

flags.DEFINE_string(
    "eval_data_output_path", None,
    "The path in which generated training input data will be written as tf"
    " records."
)

flags.DEFINE_string("meta_data_file_path", None,
                    "The path in which input meta data will be written.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")


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

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
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

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

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


def generate_tf_record_from_data_file(processor,
                                      data_dir,
                                      vocab_file,
                                      train_data_output_path=None,
                                      eval_data_output_path=None,
                                      max_seq_length=128,
                                      do_lower_case=True):
  """Generates and saves training data into a tf record file.

  Arguments:
      processor: Input processor object to be used for generating data. Subclass
        of `DataProcessor`.
      data_dir: Directory that contains train/eval data to process. Data files
        should be in from "dev.tsv", "test.tsv", or "train.tsv".
      vocab_file: Text file with words to be used for training/evaluation.
      train_data_output_path: Output to which processed tf record for training
        will be saved.
      eval_data_output_path: Output to which processed tf record for evaluation
        will be saved.
      max_seq_length: Maximum sequence length of the to be generated
        training/eval data.
      do_lower_case: Whether to lower case input text.

  Returns:
      A dictionary containing input meta data.
  """
  assert train_data_output_path or eval_data_output_path

  tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)
  assert train_data_output_path
  train_input_data_examples = processor.get_train_examples(data_dir)
  file_based_convert_examples_to_features(train_input_data_examples,
                                          max_seq_length, tokenizer,
                                          train_data_output_path)
  num_training_data = len(train_input_data_examples)

  if eval_data_output_path:
    eval_input_data_examples = processor.get_dev_examples(data_dir)
    file_based_convert_examples_to_features(eval_input_data_examples,
                                            max_seq_length, tokenizer,
                                            eval_data_output_path)

  meta_data = {
      "train_data_size": num_training_data,
      "max_seq_length": max_seq_length,
  }

  if eval_data_output_path:
    meta_data["eval_data_size"] = len(eval_input_data_examples)

  return meta_data


def generate_regression_dataset():
  assert FLAGS.input_data_dir

  processor = StsProcessor()
  return generate_tf_record_from_data_file(
      processor,
      FLAGS.input_data_dir,
      FLAGS.vocab_file,
      train_data_output_path=FLAGS.train_data_output_path,
      eval_data_output_path=FLAGS.eval_data_output_path,
      max_seq_length=FLAGS.max_seq_length,
      do_lower_case=FLAGS.do_lower_case)


def main(_):
  input_meta_data = generate_regression_dataset()

  with tf.io.gfile.GFile(FLAGS.meta_data_file_path, "w") as writer:
    writer.write(json.dumps(input_meta_data, indent=4) + "\n")


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("train_data_output_path")
  flags.mark_flag_as_required("meta_data_file_path")
  app.run(main)
