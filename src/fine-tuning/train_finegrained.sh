accelerate launch \
    --main_process_port 29600 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 2 \
    --multi_gpu \
    --mixed_precision bf16 \
    fine-tuning/train_finegrained_six_models.py \
        --config fine-tuning/fine_grained_config.yml

