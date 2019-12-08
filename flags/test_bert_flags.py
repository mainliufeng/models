from absl import flags


def define_test_bert_flags():
  flags.DEFINE_string('input_data_dir', None, 'Path to evaluation data for BERT classifier.')
  flags.DEFINE_integer('batch_size', 32, 'Batch size for evaluation.')
  flags.DEFINE_integer("seq_length", 128,
                       "The maximum total input sequence length after WordPiece tokenization. "
                       "Sequences longer than this will be truncated, and sequences shorter "
                       "than this will be padded.")

