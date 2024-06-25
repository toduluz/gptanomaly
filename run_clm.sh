DATASET_NAME=BGL
main_process_port=$(( $RANDOM % 10 + 29500 ))
echo "Main process port: $main_process_port"

NUM_SAMPLES=5000
export PYTHONPATH="${PYTHONPATH}:./"
accelerate launch --main_process_port $main_process_port scripts/run_clm_no_trainer.py \
    --train_path ../dataset/${DATASET_NAME}/train.pkl \
    --test_path ../dataset/${DATASET_NAME}/test.pkl \
    --log_template_file ../dataset/${DATASET_NAME}/${DATASET_NAME}.log_structured.csv \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 5 \
    --output_dir outputs/${DATASET_NAME} \
    --model_name_or_path gpt2 \
    --validation_split_percentage 10 \
    --checkpointing_steps epoch \
    --preprocessing_num_workers 1 \
    --max_train_samples ${NUM_SAMPLES} \
    --max_eval_samples ${NUM_SAMPLES} \
    --max_test_samples ${NUM_SAMPLES} \