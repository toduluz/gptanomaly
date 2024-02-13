DATASET_NAME=hdfs
main_process_port=$(( $RANDOM % 10 + 29500 ))
echo "Main process port: $main_process_port"

NUM_SAMPLES=50

accelerate launch --main_process_port $main_process_port scripts/run_mlm_no_trainer.py \
    --train_file data/${DATASET_NAME}/train.csv \
    --validation_file data/${DATASET_NAME}/validation.csv \
    --test_file data/${DATASET_NAME}/test.csv \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 20 \
    --output_dir outputs/${DATASET_NAME} \
    --model_name_or_path ./model \
    --trust_remote_code true \
    --tokenizer_name roberta-base \
    --pad_to_max_length \
    --line_by_line true \
    --checkpointing_steps epoch \
    --preprocessing_num_workers 10 \
    --mlm_probability 0.15 \
    --early_stopping_patience 5 \
    --max_train_samples ${NUM_SAMPLES} \
    --max_eval_samples ${NUM_SAMPLES} \
    --max_test_samples ${NUM_SAMPLES} \
