DATASET_NAME=bgl
main_process_port=$(( $RANDOM % 10 + 29500 ))
echo "Main process port: $main_process_port"
# --main_process_port $main_process_port
# STEPS=500
accelerate launch scripts/run_mlm_no_trainer.py \
    --train_file data/${DATASET_NAME}/train.csv \
    --validation_file data/${DATASET_NAME}/validation.csv \
    --test_file data/${DATASET_NAME}/test.csv \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 10 \
    --output_dir outputs/${DATASET_NAME} \
    --model_name_or_path ./model \
    --trust_remote_code true \
    --tokenizer_name roberta-base \
    --pad_to_max_length \
    --line_by_line true \
    --checkpointing_steps epoch \
    --preprocessing_num_workers 10 \
    --mlm_probability 0.15 \
    --max_train_samples 5000 \
    --max_eval_samples 5000 \
    --max_test_samples 5000 \
