from absl import flags


def define_train_bert_flags():
  flags.DEFINE_string('input_data_dir', None, '')
  flags.DEFINE_integer('train_batch_size', 32, 'Batch size for training.')
  flags.DEFINE_integer('eval_batch_size', 32, 'Batch size for evaluation.')
  flags.DEFINE_boolean('fp16', False, 'fp16')
  flags.DEFINE_string('loss', None, 'loss')
