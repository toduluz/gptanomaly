DATASET_NAME=BGL
main_process_port=$(( $RANDOM % 10 + 29500 ))
echo "Main process port: $main_process_port"

NUM_SAMPLES=50
TRANSFORMER=roberta-base
export PYTHONPATH="${PYTHONPATH}:./"
accelerate launch --main_process_port $main_process_port scripts/run.py \
    --train_path ../dataset/${DATASET_NAME}/train.pkl \
    --test_path ../dataset/${DATASET_NAME}/test.pkl \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 100 \
    --output_dir outputs/${DATASET_NAME} \
    --model_name_or_path ./model \
    --trust_remote_code true \
    --tokenizer_name ${TRANSFORMER} \
    --pad_to_max_length \
    --line_by_line true \
    --checkpointing_steps epoch \
    --preprocessing_num_workers 1 \
    --mlm_probability 0.15 \
    --early_stopping_patience 100 \
    --normal_only true \
    --encoder_name_or_path ${TRANSFORMER} \
    # --max_train_samples ${NUM_SAMPLES} \
    # --max_eval_samples ${NUM_SAMPLES} \
    # --max_test_samples ${NUM_SAMPLES} \
