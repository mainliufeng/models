export PYTHONPATH="$PYTHONPATH:/home/liufeng/Code/github/models"

python /home/liufeng/Code/github/models/data/gen_uda_dataset.py \
    --input_data_dir=$LCQMC_DIR \
    --vocab_file=${MODEL_SOURCE_DIR}/bert_chinese_L-12_H-768_A-12/vocab.txt \
    --max_seq_length=128 \
    --token_prob=0.1 \
    --index=0 \
    --classification_task_name=LCQMC_PAIR
