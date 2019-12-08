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
"""Defining common flags used across all BERT models/applications."""

from absl import flags


def define_common_bert_flags():
  """Define the flags related to TPU's."""
  flags.DEFINE_string('bert_config_file', None,
                      'Bert configuration file to define core bert layers.')
  flags.DEFINE_string('model_dir', None, (
      'The directory where the model weights and training/evaluation summaries '
      'are stored. If not specified, save to /tmp/bert20/.'))
  flags.DEFINE_string(
      'model_export_path', None,
      'Path to the directory, where trainined model will be '
      'exported.')
  flags.DEFINE_string('tpu', '', 'TPU address to connect to.')
  flags.DEFINE_string(
      'init_checkpoint', None,
      'Initial checkpoint (usually from a pre-trained BERT model).')
  flags.DEFINE_enum(
      'strategy_type', 'mirror', ['tpu', 'mirror', 'multi_worker_mirror'],
      'Distribution Strategy type to use for training. `tpu` uses '
      'TPUStrategy for running on TPUs, `mirror` uses GPUs with single host, '
      '`multi_worker_mirror` uses CPUs or GPUs with multiple hosts.')
  flags.DEFINE_float('num_train_epochs', 3.0,
                       'Total number of training epochs to perform.')
  flags.DEFINE_integer(
      'steps_per_loop', 200,
      'Number of steps per graph-mode loop. Only training step '
      'happens inside the loop. Callbacks will not be called '
      'inside.')
  flags.DEFINE_float('learning_rate', 5e-5,
                     'The initial learning rate for Adam.')
  flags.DEFINE_boolean(
      'run_eagerly', False,
      'Run the model op by op without building a model function.')
  flags.DEFINE_boolean(
      'scale_loss', False,
      'Whether to divide the loss by number of replica inside the per-replica '
      'loss function.')
  flags.DEFINE_boolean(
      'use_keras_compile_fit', True,
      'If True, uses Keras compile/fit() API for training logic. Otherwise '
      'use custom training loop.')
  flags.DEFINE_string(
      'hub_module_url', None, 'TF-Hub path/url to Bert module. '
      'If specified, init_checkpoint flag should not be used.')
  flags.DEFINE_boolean(
      'share_parameter_across_layers', False,
      'If True, share parameters across layers.')
