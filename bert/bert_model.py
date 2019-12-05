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
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from modeling.bert import BertModel


def get_bert_model(input_word_ids,
                   input_mask,
                   input_type_ids,
                   config=None,
                   name=None,
                   float_type=tf.float32,
                   share_parameter_across_layers=False):
  """Wraps the core BERT model as a keras.Model."""
  bert_model_layer = BertModel(
    config=config, float_type=float_type,
    share_parameter_across_layers=share_parameter_across_layers, name=name)
  pooled_output, sequence_output = bert_model_layer(input_word_ids, input_mask,
                                                    input_type_ids)
  bert_model = tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=[pooled_output, sequence_output])
  return bert_model


