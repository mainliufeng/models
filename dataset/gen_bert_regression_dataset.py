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

from absl import flags, app

from data.bert_regression_data_lib import StsProcessor
from dataset.bert_regression_dataset import BertRegressionDataset

FLAGS = flags.FLAGS

# BERT classification specific flags.
flags.DEFINE_string(
    "input_data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

# Shared flags across BERT fine-tuning tasks.
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")


def main(_):
  dataset = BertRegressionDataset(FLAGS.input_data_dir)
  dataset.gen_tf_records(StsProcessor(), FLAGS.vocab_file, FLAGS.max_seq_length)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_data_dir")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("max_seq_length")
  app.run(main)
