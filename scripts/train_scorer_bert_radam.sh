export PYTHONPATH="$PYTHONPATH:/home/liufeng/Code/github/models"

python /home/liufeng/Code/github/models/bin/train_regression_by_fit.py \
  --input_data_dir=$DATASET_DIR/scorer_simi \
  --bert_config_file=${MODEL_SOURCE_DIR}/bert_chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=${MODEL_SOURCE_DIR}/tf2_bert_chinese_L-12_H-768_A-12/bert_model.ckpt \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=6 \
  --optimizer=radam \
  --model_dir=$MODEL_DIR/scorer_simi/bert/tf2/opt-radam_lr-2e-5_batch-32_epoch-6
