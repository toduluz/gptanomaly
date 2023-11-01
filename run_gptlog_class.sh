DATASET_NAME=bgl
STEPS=500
accelerate launch scripts/run_classification.py \
    --label_column_name labels \
    --text_column_names text \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file data/${DATASET_NAME}/train.csv \
    --validation_file data/${DATASET_NAME}/validation.csv \
    --test_file data/${DATASET_NAME}/test.csv \
    --metric_name f1 \
    --model_name_or_path gpt2-medium \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_step 1 \
    --per_device_eval_batch_size 16 \
    --output_dir outputs/${DATASET_NAME} \
    --overwrite_cache \
    --save_total_limit 5 \
    --eval_steps ${STEPS} \
    --save_steps ${STEPS} \
    --logging_steps ${STEPS} \
    --greater_is_better True \
    --metric_for_best_model f1 \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --overwrite_output_dir \
    --max_steps 50000 \