export PYTHONPATH="$PYTHONPATH:/home/liufeng/Code/github/models"

python /home/liufeng/Code/github/models/bin/test_regression.py \
  --input_data_dir=$DATASET_DIR/scorer_simi \
  --bert_config_file=${MODEL_SOURCE_DIR}/bert_chinese_L-12_H-768_A-12/bert_config.json \
  --batch_size=32 \
  --model_dir=$MODEL_DIR/scorer_simi/bert/tf2/lr-loss-huber_2e-5_batch-32_epoch-3
