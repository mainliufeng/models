export PYTHONPATH="$PYTHONPATH:/home/liufeng/Code/github/models"

python /home/liufeng/Code/github/models/bin/run_bert_tf1_to_keras_checkpoint_convert.py \
  --checkpoint_from_path=${MODEL_SOURCE_DIR}/albert_tiny_250k/albert_model.ckpt \
  --checkpoint_to_path=${MODEL_SOURCE_DIR}/tf2_albert_tiny_250k/albert_model_temp.ckpt
python /home/liufeng/Code/github/models/bin/run_bert_tf2_checkpoint_convert.py \
  --bert_config_file=${MODEL_SOURCE_DIR}/albert_tiny_250k/albert_config_tiny.json \
  --init_checkpoint=${MODEL_SOURCE_DIR}/tf2_albert_tiny_250k/albert_model_temp.ckpt \
  --share_parameter_across_layers=true \
  --converted_checkpoint=${MODEL_SOURCE_DIR}/tf2_albert_tiny_250k/albert_model.ckpt
rm ${MODEL_SOURCE_DIR}/tf2_albert_tiny_250k/albert_model_temp.ckpt*
