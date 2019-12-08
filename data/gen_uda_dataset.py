# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
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
"""Preprocessing for text classifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import math
import collections
from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf
import jieba
import json
import string

from data.bert_classifier_data_lib import convert_single_example, LCQMCPairClassificationProcessor
from data import bert_tokenization as tokenization


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



printable = set(string.printable)


def _filter_unicode(st):
  return "".join([c for c in st if c in printable])


def generate_tf_record_from_data_file(processor,
                                      data_dir,
                                      vocab_file,
                                      token_prob,
                                      index,
                                      max_seq_length=128,
                                      do_lower_case=True):
  label_list = processor.get_labels()
  tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)


  assert data_dir
  train_input_data_examples = processor.get_examples(data_dir, 'train')

  data_stats_dir = os.path.join(data_dir, "data_stats")
  unsup_out_dir = os.path.join(
    data_dir, "unsup", "tf_idf-{}".format(token_prob), str(index))
  _proc_and_save_unsup_data(
    train_input_data_examples, processor.get_labels(), data_stats_dir,
    unsup_out_dir, tokenizer, max_seq_length, token_prob)


def _proc_and_save_unsup_data(
  ori_examples, sup_labels, data_stats_dir, unsup_out_dir, tokenizer,
  max_seq_length, token_prob, max_shard_size=4096):
  random_seed = np.random.randint(0, 100000)
  logging.info("random seed: {:d}".format(random_seed))
  np.random.seed(random_seed)
  logging.info("getting examples")

  logging.info("getting augmented examples")
  aug_examples = copy.deepcopy(ori_examples)

  labels = sup_labels + ["unsup"]
  logging.info("processing ori examples")
  ori_examples = _tokenize_examples(ori_examples, tokenizer)
  ori_features = [
    convert_single_example(ex_index, ori_example, labels, max_seq_length, tokenizer)
    for ex_index, ori_example in enumerate(ori_examples)]

  logging.info("processing aug examples")
  aug_examples = _tokenize_examples(aug_examples, tokenizer)
  data_stats = _get_data_stats(data_stats_dir, ori_examples)
  op = TfIdfWordRep(token_prob, data_stats)
  for i in range(len(aug_examples)):
    aug_examples[i] = op(aug_examples[i])
    aug_examples[i].text_a = ''.join(aug_examples[i].word_list_a)
    aug_examples[i].text_b = ''.join(aug_examples[i].word_list_b)
  aug_features = [
    convert_single_example(ex_index, aug_example, labels, max_seq_length, tokenizer)
    for ex_index, aug_example in enumerate(aug_examples)]

  for ori_example, aug_example in zip(ori_examples[:5], aug_examples[:5]):
    logging.info("{} -> {}".format(ori_example.text_a, aug_example.text_a))
    logging.info("{} -> {}".format(ori_example.text_b, aug_example.text_b))

  unsup_features = []
  for ori_feat, aug_feat in zip(ori_features, aug_features):
    unsup_features.append(PairedUnsupInputFeatures(
        ori_feat.input_ids,
        ori_feat.input_mask,
        ori_feat.segment_ids,
        aug_feat.input_ids,
        aug_feat.input_mask,
        aug_feat.segment_ids,
        ))

  """Dump tf record."""
  if not tf.io.gfile.exists(unsup_out_dir):
    tf.io.gfile.makedirs(unsup_out_dir)
  logging.info("dumping TFRecords")
  np.random.shuffle(unsup_features)
  shard_cnt = 0
  shard_size = 0
  tfrecord_writer = _obtain_tfrecord_writer(unsup_out_dir, shard_cnt)
  for feature in unsup_features:
    tf_example = tf.train.Example(
        features=tf.train.Features(feature=feature.get_dict_features()))
    if shard_size >= max_shard_size:
      tfrecord_writer.close()
      shard_cnt += 1
      tfrecord_writer = _obtain_tfrecord_writer(unsup_out_dir, shard_cnt)
      shard_size = 0
    shard_size += 1
    tfrecord_writer.write(tf_example.SerializeToString())
  tfrecord_writer.close()


def _get_tf_idf_by_word_list(examples):
  """通过example中的word_list_a和word_list_b计算tf-idf信息"""
  word_doc_freq = collections.defaultdict(int)
  # Compute IDF
  for i in range(len(examples)):
    cur_word_dict = {}
    cur_sent = copy.deepcopy(examples[i].word_list_a)
    if examples[i].text_b:
      cur_sent += examples[i].word_list_b
    for word in cur_sent:
      cur_word_dict[word] = 1
    for word in cur_word_dict:
      word_doc_freq[word] += 1
  idf = {}
  for word in word_doc_freq:
    idf[word] = math.log(len(examples) * 1. / word_doc_freq[word])
  # Compute TF-IDF
  tf_idf = {}
  for i in range(len(examples)):
    cur_word_dict = {}
    cur_sent = copy.deepcopy(examples[i].word_list_a)
    if examples[i].text_b:
      cur_sent += examples[i].word_list_b
    for word in cur_sent:
      if word not in tf_idf:
        tf_idf[word] = 0
      tf_idf[word] += 1. / len(cur_sent) * idf[word]
  return {
      "idf": idf,
      "tf_idf": tf_idf,
  }


def _get_data_stats(data_stats_dir, examples):
  keys = ["tf_idf", "idf"]
  all_exist = True
  for key in keys:
    data_stats_path = "{}/{}.json".format(data_stats_dir, key)
    if not tf.io.gfile.exists(data_stats_path):
      all_exist = False
      logging.info("Not exist: {}".format(data_stats_path))
  if all_exist:
    logging.info("loading data stats from {:s}".format(data_stats_dir))
    data_stats = {}
    for key in keys:
      with tf.io.gfile.GFile(
          "{}/{}.json".format(data_stats_dir, key), "r") as inf:
        data_stats[key] = json.load(inf)
  else:
    data_stats = _get_tf_idf_by_word_list(examples)
    tf.io.gfile.makedirs(data_stats_dir)
    for key in keys:
      with tf.io.gfile.GFile("{}/{}.json".format(data_stats_dir, key), "w") as ouf:
        json.dump(data_stats[key], ouf)
    logging.info("dumped data stats to {:s}".format(data_stats_dir))
  return data_stats


def _tokenize_examples(examples, tokenizer):
  """jieba分词"""
  logging.info("tokenizing examples")
  for i in range(len(examples)):
    examples[i].word_list_a = list(jieba.cut(examples[i].text_a, cut_all=False))
    if examples[i].text_b:
      examples[i].word_list_b = list(jieba.cut(examples[i].text_b, cut_all=False))
    if i % 10000 == 0:
      logging.info("finished tokenizing example {:d}".format(i))
  return examples


def _create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


class PairedUnsupInputFeatures(object):
  """Features for paired unsup data."""

  def __init__(self, ori_input_ids, ori_input_mask, ori_segment_ids,
               aug_input_ids, aug_input_mask, aug_segment_ids):
    self.ori_input_ids = ori_input_ids
    self.ori_input_mask = ori_input_mask
    self.ori_segment_ids = ori_segment_ids
    self.aug_input_ids = aug_input_ids
    self.aug_input_mask = aug_input_mask
    self.aug_segment_ids = aug_segment_ids

  def get_dict_features(self):
    return {
        "ori_input_ids": _create_int_feature(self.ori_input_ids),
        "ori_input_mask": _create_int_feature(self.ori_input_mask),
        "ori_segment_ids": _create_int_feature(self.ori_segment_ids),
        "aug_input_ids": _create_int_feature(self.aug_input_ids),
        "aug_input_mask": _create_int_feature(self.aug_input_mask),
        "aug_segment_ids": _create_int_feature(self.aug_segment_ids),
    }


def _obtain_tfrecord_writer(data_path, shard_cnt):
  tfrecord_writer = tf.io.TFRecordWriter(
      os.path.join(
          data_path,
          "tf_examples.tfrecord.{:d}".format(shard_cnt)))
  return tfrecord_writer


class EfficientRandomGen(object):
  """A base class that generate multiple random numbers at the same time."""

  def reset_random_prob(self):
    """Generate many random numbers at the same time and cache them."""
    cache_len = 100000
    self.random_prob_cache = np.random.random(size=(cache_len,))
    self.random_prob_ptr = cache_len - 1

  def get_random_prob(self):
    """Get a random number."""
    value = self.random_prob_cache[self.random_prob_ptr]
    self.random_prob_ptr -= 1
    if self.random_prob_ptr == -1:
      self.reset_random_prob()
    return value

  def get_random_token(self):
    """Get a random token."""
    token = self.token_list[self.token_ptr]
    self.token_ptr -= 1
    if self.token_ptr == -1:
      self.reset_token_list()
    return token


class TfIdfWordRep(EfficientRandomGen):
  """TF-IDF Based Word Replacement."""

  def __init__(self, token_prob, data_stats):
    super(TfIdfWordRep, self).__init__()
    self.token_prob = token_prob
    self.data_stats = data_stats
    self.idf = data_stats["idf"]
    self.tf_idf = data_stats["tf_idf"]
    data_stats = copy.deepcopy(data_stats)
    tf_idf_items = data_stats["tf_idf"].items()
    tf_idf_items = sorted(tf_idf_items, key=lambda item: -item[1])
    self.tf_idf_keys = []
    self.tf_idf_values = []
    for key, value in tf_idf_items:
      self.tf_idf_keys += [key]
      self.tf_idf_values += [value]
    self.normalized_tf_idf = np.array(self.tf_idf_values)
    self.normalized_tf_idf = (self.normalized_tf_idf.max()
                              - self.normalized_tf_idf)
    self.normalized_tf_idf = (self.normalized_tf_idf
                              / self.normalized_tf_idf.sum())
    self.reset_token_list()
    self.reset_random_prob()

  def get_replace_prob(self, all_words):
    """Compute the probability of replacing tokens in a sentence."""
    cur_tf_idf = collections.defaultdict(int)
    for word in all_words:
      cur_tf_idf[word] += 1. / len(all_words) * self.idf[word]
    replace_prob = []
    for word in all_words:
      replace_prob += [cur_tf_idf[word]]
    replace_prob = np.array(replace_prob)
    replace_prob = np.max(replace_prob) - replace_prob
    replace_prob = (replace_prob / replace_prob.sum() *
                    self.token_prob * len(all_words))
    return replace_prob

  def __call__(self, example):
    if self.get_random_prob() < 0.001:
      show_example = True
    else:
      show_example = False
    all_words = copy.deepcopy(example.word_list_a)
    if example.text_b:
      all_words += example.word_list_b

    if show_example:
      logging.info("before tf_idf_unif aug: {:s}".format(
          _filter_unicode(" ".join(all_words))))

    replace_prob = self.get_replace_prob(all_words)
    example.word_list_a = self.replace_tokens(
        example.word_list_a,
        replace_prob[:len(example.word_list_a)]
        )
    if example.text_b:
      example.word_list_b = self.replace_tokens(
          example.word_list_b,
          replace_prob[len(example.word_list_a):]
          )

    if show_example:
      all_words = copy.deepcopy(example.word_list_a)
      if example.text_b:
        all_words += example.word_list_b
      logging.info("after tf_idf_unif aug: {:s}".format(
          _filter_unicode(" ".join(all_words))))
    return example

  def replace_tokens(self, word_list, replace_prob):
    """Replace tokens in a sentence."""
    for i in range(len(word_list)):
      if self.get_random_prob() < replace_prob[i]:
        word_list[i] = self.get_random_token()
    return word_list

  def reset_token_list(self):
    cache_len = len(self.tf_idf_keys)
    token_list_idx = np.random.choice(
        cache_len, (cache_len,), p=self.normalized_tf_idf)
    self.token_list = []
    for idx in token_list_idx:
      self.token_list += [self.tf_idf_keys[idx]]
    self.token_ptr = len(self.token_list) - 1


def main(_):
  """Generates classifier dataset and returns input meta data."""
  assert FLAGS.input_data_dir and FLAGS.classification_task_name

  processors = {
      "lcqmc_pair": LCQMCPairClassificationProcessor,
  }
  task_name = FLAGS.classification_task_name.lower()
  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()
  return generate_tf_record_from_data_file(
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
