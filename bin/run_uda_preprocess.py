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
"""BERT finetuning task dataset generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from absl import app
from absl import flags
import tensorflow as tf

from data import uda_data_lib
from data import bert_classifier_data_lib as classifier_data_lib

FLAGS = flags.FLAGS

# BERT classification specific flags.
flags.DEFINE_string(
    "input_data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_enum("classification_task_name", "MNLI",
                  ["COLA", "MNLI", "MRPC", "XNLI", "ATEC", "SIM", "LCQMC_PAIR"],
                  "The name of the task to train BERT classifier.")

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

flags.DEFINE_float("token_prob", 0.7, "Token prob")
flags.DEFINE_integer("index", 0, "index")


def main(_):
  """Generates classifier dataset and returns input meta data."""
  assert FLAGS.input_data_dir and FLAGS.classification_task_name

  processors = {
      "cola": classifier_data_lib.ColaProcessor,
      "mnli": classifier_data_lib.MnliProcessor,
      "mrpc": classifier_data_lib.MrpcProcessor,
      "xnli": classifier_data_lib.XnliProcessor,
      "atec": classifier_data_lib.AtecProcessor,
      "lcqmc_pair": classifier_data_lib.LCQMCPairClassificationProcessor,
  }
  task_name = FLAGS.classification_task_name.lower()
  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()
  return uda_data_lib.generate_tf_record_from_data_file(
      processor,
      FLAGS.input_data_dir,
      FLAGS.vocab_file,
      FLAGS.token_prob,
      FLAGS.index,
      max_seq_length=FLAGS.max_seq_length,
      do_lower_case=FLAGS.do_lower_case)


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("input_data_dir")
  app.run(main)
