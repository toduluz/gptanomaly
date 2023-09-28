accelerate launch scripts/run_clm_no_trainer.py \
    --train_file data/bgl/train.csv \
    --validation_file data/bgl/test.csv \
    --model_name_or_path gpt2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_train_steps 1000 \
    --output_dir outputs \
    --preprocessing_num_workers 5 \
    --overwrite_cache \
    --seed 123 \
    --max_eval_samples 100 \
    # --num_beams 2 \
