export PYTHONPATH="$PYTHONPATH:/home/liufeng/Code/github/models"

python /home/liufeng/Code/github/models/bin/train_regression.py \
  --input_data_dir=$DATASET_DIR/scorer_simi_train_dev \
  --bert_config_file=${MODEL_SOURCE_DIR}/bert_chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=${MODEL_DIR}/easemob_dialog/201811_test_continue_epoch-3_lr-2e-5/checkpoint \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --model_dir=$MODEL_DIR/scorer_simi_train_dev/bert/tf2/lr-2e-5_batch-32_epoch-3_sigmoid_easemob_dialog
