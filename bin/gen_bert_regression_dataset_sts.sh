export PYTHONPATH="$PYTHONPATH:/home/liufeng/Code/github/models"

python /home/liufeng/Code/github/models/data/gen_bert_regression_dataset.py \
    --input_data_dir=$DATASET_DIR/scorer_simi \
    --vocab_file=${MODEL_SOURCE_DIR}/bert_chinese_L-12_H-768_A-12/vocab.txt \
    --train_data_output_path=$DATASET_DIR/scorer_simi/train.tf_record \
    --eval_data_output_path=$DATASET_DIR/scorer_simi/eval.tf_record \
    --meta_data_file_path=$DATASET_DIR/scorer_simi/meta_data \
    --max_seq_length=128
