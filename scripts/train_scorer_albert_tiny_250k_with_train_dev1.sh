export PYTHONPATH="$PYTHONPATH:/home/liufeng/Code/github/models"

python /home/liufeng/Code/github/models/bin/train_regression.py \
  --input_data_dir=$DATASET_DIR/scorer_simi_train_dev1 \
  --bert_config_file=${MODEL_SOURCE_DIR}/albert_tiny_250k/albert_config_tiny.json \
  --init_checkpoint=${MODEL_SOURCE_DIR}/tf2_albert_tiny_250k/albert_model.ckpt \
  --train_batch_size=128 \
  --eval_batch_size=64 \
  --learning_rate=2e-4 \
  --num_train_epochs=5 \
  --share_parameter_across_layers=true \
  --model_dir=$MODEL_DIR/scorer_simi_train_dev1/albert/tf2/lr-2e-4_batch-128_epoch-5_sigmoid
