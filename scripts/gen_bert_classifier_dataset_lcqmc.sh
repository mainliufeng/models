export PYTHONPATH="$PYTHONPATH:/home/liufeng/Code/github/models"

export TASK_NAME=LCQMC_PAIR

python /home/liufeng/Code/github/models/data/gen_bert_classifier_dataset.py \
    --input_data_dir=$LCQMC_DIR \
    --vocab_file=${MODEL_SOURCE_DIR}/bert_chinese_L-12_H-768_A-12/vocab.txt \
    --train_data_output_path=$LCQMC_DIR/${TASK_NAME}_train.tf_record \
    --eval_data_output_path=$LCQMC_DIR/${TASK_NAME}_eval.tf_record \
    --meta_data_file_path=$LCQMC_DIR/${TASK_NAME}_meta_data \
    --max_seq_length=128 \
    --classification_task_name=${TASK_NAME}
