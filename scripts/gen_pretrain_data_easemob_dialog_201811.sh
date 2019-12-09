export PYTHONPATH="$PYTHONPATH:/home/liufeng/Code/github/models"

python /home/liufeng/Code/github/models/data/create_pretraining_data.py \
  --input_file=$DATASET_DIR/easemob_dialog/201811_test/* \
  --output_file=$DATASET_DIR/easemob_dialog/201811_test.tf_record \
  --vocab_file=${MODEL_SOURCE_DIR}/bert_chinese_L-12_H-768_A-12/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
