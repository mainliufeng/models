export PYTHONPATH="$PYTHONPATH:/home/liufeng/Code/github/models"

python /home/liufeng/Code/github/models/bin/export_classifier.py \
  --input_meta_data_path=$LCQMC_DIR/meta_data \
  --bert_config_file=${MODEL_SOURCE_DIR}/albert_tiny_250k/albert_config_tiny.json \
  --share_parameter_across_layers=true \
  --model_dir=$MODEL_DIR/lcqmc/albert/tf2/lr-1e-4_batch-64_epoch-5 \
  --model_export_path=$MODEL_DIR/lcqmc/albert/tf2/lr-1e-4_batch-64_epoch-5/export
