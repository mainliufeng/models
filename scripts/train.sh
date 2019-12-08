export PYTHONPATH="$PYTHONPATH:/home/liufeng/Code/github/models"

rm -rf $MODEL_DIR/lcqmc/albert/tf2/train
python /home/liufeng/Code/github/models/bin/train_classifier_by_fit.py \
  --input_data_dir=$LCQMC_DIR \
  --bert_config_file=${MODEL_SOURCE_DIR}/albert_tiny_250k/albert_config_tiny.json \
  --init_checkpoint=${MODEL_SOURCE_DIR}/tf2_albert_tiny_250k/albert_model.ckpt \
  --train_batch_size=1 \
  --eval_batch_size=1 \
  --learning_rate=1e-4 \
  --num_train_epochs=5 \
  --share_parameter_across_layers=true \
  --model_dir=$MODEL_DIR/lcqmc/albert/tf2/train
