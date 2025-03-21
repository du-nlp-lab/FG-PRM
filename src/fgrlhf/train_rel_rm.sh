set -e

# --nproc_per_node=gpu --standalone
# --nproc_per_node 1 --standalone --nnodes=1

# train reward model for NF-ERR_subsentence
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=gpu --standalone ./reward_modeling/run_fg_rm.py \
                --model_name_or_path /data/models/huggingface-format/llama-3-8b-instruct \
                --train_file ./tasks/qa_feedback/data/NF-ERR_subsentence/train.json \
                --validation_file ./tasks/qa_feedback/data/NF-ERR_subsentence/dev.json \
                --test_file ./tasks/qa_feedback/data/NF-ERR_subsentence/dev.json \
                --output_dir ./tasks/qa_feedback/model_outputs/rel_rm \
                --do_train \
                --do_eval \
                --do_predict \
                --bf16 \
                --num_train_epochs 3 \
                --per_device_train_batch_size 1 \
                --per_device_eval_batch_size 1 \
                --evaluation_strategy epoch \
                --logging_strategy epoch \
                --save_strategy epoch \
                --load_best_model_at_end \
                --metric_for_best_model overall_accuracy \
                --max_seq_length 2048 \
                --report_to wandb \
                --save_total_limit 2 \
                --learning_rate 0.000005 \
                --weight_decay 0.01 \
                --warmup_ratio 0.1

