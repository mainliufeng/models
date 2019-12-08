export PYTHONPATH="$PYTHONPATH:/home/liufeng/Code/github/models"

python /home/liufeng/Code/github/models/bin/test_classifier.py \
  --input_data_dir=$DATASET_DIR/LCQMC \
  --bert_config_file=${MODEL_SOURCE_DIR}/albert_tiny_250k/albert_config_tiny.json \
  --batch_size=128 \
  --share_parameter_across_layers=true \
  --model_dir=$MODEL_DIR/lcqmc/albert/tf2/lr-1e-4_batch-64_epoch-10_focal
