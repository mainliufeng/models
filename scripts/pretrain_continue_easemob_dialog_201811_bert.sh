export PYTHONPATH="$PYTHONPATH:/home/liufeng/Code/github/models"

python /home/liufeng/Code/github/models/bin/pretrain.py \
  --input_files=$DATASET_DIR/easemob_dialog/201811_test.tf_record \
  --model_dir=$MODEL_DIR/easemob_dialog/201811_test_continue_epoch-3_lr-2e-5 \
  --bert_config_file=${MODEL_SOURCE_DIR}/bert_chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=${MODEL_SOURCE_DIR}/tf2_bert_chinese_L-12_H-768_A-12/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_epochs=20 \
  --num_steps_per_epoch=1000 \
  --warmup_steps=10000 \
  --learning_rate=2e-5
