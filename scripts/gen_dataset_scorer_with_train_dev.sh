export PYTHONPATH="$PYTHONPATH:/home/liufeng/Code/github/models"

python /home/liufeng/Code/github/models/dataset/gen_bert_regression_dataset_with_train_dev.py \
    --input_data_dir=$DATASET_DIR/scorer_simi_train_dev \
    --vocab_file=${MODEL_SOURCE_DIR}/bert_chinese_L-12_H-768_A-12/vocab.txt \
    --max_seq_length=128
