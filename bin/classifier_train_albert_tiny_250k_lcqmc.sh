export PYTHONPATH="$PYTHONPATH:/home/liufeng/Code/github/models"

export TASK=LCQMC_PAIR

python /home/liufeng/Code/github/models/bin/run_bert_classifier_train.py \
  --input_meta_data_path=$LCQMC_DIR/${TASK}_meta_data \
  --train_data_path=$LCQMC_DIR/${TASK}_train.tf_record \
  --eval_data_path=$LCQMC_DIR/${TASK}_eval.tf_record \
  --bert_config_file=${MODEL_SOURCE_DIR}/albert_tiny_250k/albert_config_tiny.json \
  --init_checkpoint=${MODEL_SOURCE_DIR}/tf2_albert_tiny_250k/albert_model.ckpt \
  --train_batch_size=64 \
  --eval_batch_size=64 \
  --learning_rate=1e-4 \
  --num_train_epochs=5 \
  --share_parameter_across_layers=true \
  --model_dir=$MODEL_DIR/lcqmc/albert/tf2/lr-1e-4_batch-64_epoch-5
