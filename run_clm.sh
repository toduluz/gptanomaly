DATASET_NAME=HDFS
main_process_port=$(( $RANDOM % 10 + 29500 ))
echo "Main process port: $main_process_port"

NUM_SAMPLES=50
TRANSFORMER=gpt2
export PYTHONPATH="${PYTHONPATH}:./"
accelerate launch --main_process_port $main_process_port scripts/run_clm_no_trainer.py \
    --train_path ../dataset/${DATASET_NAME}/train.pkl \
    --test_path ../dataset/${DATASET_NAME}/test.pkl \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 10 \
    --output_dir outputs/${DATASET_NAME} \
    --model_name_or_path gpt2 \
    --baseline_model_name_or_path ${TRANSFORMER} \
    --validation_split_percentage 10 \
    --checkpointing_steps epoch \
    --preprocessing_num_workers 1 \
    --max_train_samples ${NUM_SAMPLES} \
    --max_eval_samples ${NUM_SAMPLES} \
    --max_test_samples ${NUM_SAMPLES} \