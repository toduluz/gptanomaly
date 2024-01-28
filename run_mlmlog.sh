DATASET_NAME=hdfs
STEPS=500
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
    --max_train_samples 100 \
    --max_eval_samples 100 \
    --max_test_samples 100 \
    --checkpointing_steps epoch \
    --preprocessing_num_workers 10 \
