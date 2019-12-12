export PYTHONPATH="$PYTHONPATH:/home/liufeng/Code/github/models"

python /home/liufeng/Code/github/models/data/get_unk.py \
  --input_data_dir=$DATASET_DIR/scorer_simi \
  --set_type=train \
  --vocab_file=${MODEL_SOURCE_DIR}/bert_chinese_L-12_H-768_A-12/vocab.txt
