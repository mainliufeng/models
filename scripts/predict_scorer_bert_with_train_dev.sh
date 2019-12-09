export PYTHONPATH="$PYTHONPATH:/home/liufeng/Code/github/models"

python /home/liufeng/Code/github/models/bin/predict_regression_with_train_dev.py \
  --input_data_dir=$DATASET_DIR/scorer_simi_train_dev \
  --bert_config_file=${MODEL_SOURCE_DIR}/bert_chinese_L-12_H-768_A-12/bert_config.json \
  --batch_size=32 \
  --model_dir=$MODEL_DIR/scorer_simi_train_dev/bert/tf2/lr-2e-5_batch-32_epoch-3_sigmoid
