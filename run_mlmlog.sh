DATASET_NAME=HDFS
main_process_port=$(( $RANDOM % 10 + 29500 ))
echo "Main process port: $main_process_port"

NUM_SAMPLES=500
export PYTHONPATH="${PYTHONPATH}:./"
accelerate launch --main_process_port $main_process_port scripts/run.py \
    --train_path ../dataset/${DATASET_NAME}/train.pkl \
    --test_path ../dataset/${DATASET_NAME}/test.pkl \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 10 \
    --output_dir outputs/${DATASET_NAME} \
    --model_name_or_path ./model \
    --trust_remote_code true \
    --tokenizer_name roberta-base \
    --pad_to_max_length \
    --line_by_line true \
    --checkpointing_steps epoch \
    --preprocessing_num_workers 1 \
    --mlm_probability 0.15 \
    --early_stopping_patience 100 \
    --normal_only true \
    --encoder_name_or_path roberta-base \
    --max_train_samples ${NUM_SAMPLES} \
    --max_eval_samples ${NUM_SAMPLES} \
    --max_test_samples ${NUM_SAMPLES} \
