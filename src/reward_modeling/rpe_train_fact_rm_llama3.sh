set -e

gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

# train reward model for ORM
torchrun \
    --nproc_per_node $gpu_num \
    --nnodes=1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    ./reward_modeling/run_fg_rm_llama3.py \
                --model_name_or_path /data/models/huggingface-format/llama-3-8b/ \
                --train_file ../data/hallucination_sample/train_ori_orm_llama-3-70b-instruct.json \
                --validation_file ../data/hallucination_sample/dev_ori_orm_llama-3-70b-instruct.json \
                --test_file ../data/hallucination_sample/dev_ori_orm_llama-3-70b-instruct.json \
                --output_dir ../models/orm_llama3_ori_6/ \
                --do_train \
                --do_eval \
                --do_predict \
                --bf16 \
                --num_train_epochs 50 \
                --per_device_train_batch_size 2 \
                --per_device_eval_batch_size 2 \
                --eval_strategy epoch \
                --logging_strategy epoch \
                --save_strategy epoch \
                --load_best_model_at_end \
                --metric_for_best_model overall_accuracy \
                --max_seq_length 1536 \
                --report_to wandb \
                --save_total_limit 2 \
                --learning_rate 0.000005 \
                --weight_decay 0.001 \
                --warmup_ratio 0.1

# train reward model for PRM
torchrun \
    --nproc_per_node $gpu_num \
    --nnodes=1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    ./reward_modeling/run_fg_rm_llama3.py \
                --model_name_or_path /data/models/huggingface-format/llama-3-8b/ \
                --train_file ../data/hallucination_sample/train_ori_llama-3-70b-instruct.json \
                --validation_file ../data/hallucination_sample/dev_ori_llama-3-70b-instruct.json \
                --test_file ../data/hallucination_sample/dev_ori_llama-3-70b-instruct.json \
                --output_dir ../models/prm_llama3_ori_6/ \
                --do_train \
                --do_eval \
                --do_predict \
                --bf16 \
                --num_train_epochs 50 \
                --per_device_train_batch_size 1 \
                --per_device_eval_batch_size 1 \
                --eval_strategy epoch \
                --logging_strategy epoch \
                --save_strategy epoch \
                --load_best_model_at_end \
                --metric_for_best_model overall_accuracy \
                --max_seq_length 1536 \
                --report_to wandb \
                --save_total_limit 2 \
                --learning_rate 0.000005 \
                --weight_decay 0.001 \
                --warmup_ratio 0.1

# train reward model for PRM
torchrun \
    --nproc_per_node $gpu_num \
    --nnodes=1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    ./reward_modeling/run_fg_rm_llama3.py \
        --model_name_or_path /data/models/huggingface-format/llama-3-8b/ \
        --train_file ../data/hallucination_sample/synthetic_data_669/synthetic_Calculation-Error_llama-3-70b-instruct_train.json \
        --validation_file ../data/hallucination_sample/synthetic_data_669/synthetic_Calculation-Error_llama-3-70b-instruct_dev.json \
        --test_file ../data/hallucination_sample/synthetic_data_669/synthetic_Calculation-Error_llama-3-70b-instruct_dev.json \
        --output_dir ../models/prm_llama3_Calculation-Error_6/ \
        --do_train \
        --do_eval \
        --do_predict \
        --bf16 \
        --num_train_epochs 20 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --eval_strategy epoch \
        --logging_strategy epoch \
        --save_strategy epoch \
        --load_best_model_at_end \
        --metric_for_best_model overall_accuracy \
        --max_seq_length 1536 \
        --report_to wandb \
        --save_total_limit 2 \
        --learning_rate 0.000005 \
        --weight_decay 0.001 \
        --warmup_ratio 0.1

# train reward model for PRM
torchrun \
    --nproc_per_node $gpu_num \
    --nnodes=1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    ./reward_modeling/run_fg_rm_llama3.py \
        --model_name_or_path /data/models/huggingface-format/llama-3-8b/ \
        --train_file ../data/hallucination_sample/synthetic_data_669/synthetic_Logical-Inconsistency_llama-3-70b-instruct_train.json \
        --validation_file ../data/hallucination_sample/synthetic_data_669/synthetic_Logical-Inconsistency_llama-3-70b-instruct_dev.json \
        --test_file ../data/hallucination_sample/synthetic_data_669/synthetic_Logical-Inconsistency_llama-3-70b-instruct_dev.json \
        --output_dir ../models/prm_llama3_Logical-Inconsistency_6/ \
        --do_train \
        --do_eval \
        --do_predict \
        --bf16 \
        --num_train_epochs 20 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --eval_strategy epoch \
        --logging_strategy epoch \
        --save_strategy epoch \
        --load_best_model_at_end \
        --metric_for_best_model overall_accuracy \
        --max_seq_length 1536 \
        --report_to wandb \
        --save_total_limit 2 \
        --learning_rate 0.000005 \
        --weight_decay 0.001 \
        --warmup_ratio 0.1

# train reward model for PRM
torchrun \
    --nproc_per_node $gpu_num \
    --nnodes=1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    ./reward_modeling/run_fg_rm_llama3.py \
        --model_name_or_path /data/models/huggingface-format/llama-3-8b/ \
        --train_file ../data/hallucination_sample/synthetic_data_669/synthetic_Factual-Inconsistency_llama-3-70b-instruct_train.json \
        --validation_file ../data/hallucination_sample/synthetic_data_669/synthetic_Factual-Inconsistency_llama-3-70b-instruct_dev.json \
        --test_file ../data/hallucination_sample/synthetic_data_669/synthetic_Factual-Inconsistency_llama-3-70b-instruct_dev.json \
        --output_dir ../models/prm_llama3_Factual-Inconsistency_6/ \
        --do_train \
        --do_eval \
        --do_predict \
        --bf16 \
        --num_train_epochs 20 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --eval_strategy epoch \
        --logging_strategy epoch \
        --save_strategy epoch \
        --load_best_model_at_end \
        --metric_for_best_model overall_accuracy \
        --max_seq_length 1536 \
        --report_to wandb \
        --save_total_limit 2 \
        --learning_rate 0.000005 \
        --weight_decay 0.001 \
        --warmup_ratio 0.1

# train reward model for PRM
torchrun \
    --nproc_per_node $gpu_num \
    --nnodes=1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    ./reward_modeling/run_fg_rm_llama3.py \
        --model_name_or_path /data/models/huggingface-format/llama-3-8b/ \
        --train_file ../data/hallucination_sample/synthetic_data_669/synthetic_Context-Inconsistency_llama-3-70b-instruct_train.json \
        --validation_file ../data/hallucination_sample/synthetic_data_669/synthetic_Context-Inconsistency_llama-3-70b-instruct_dev.json \
        --test_file ../data/hallucination_sample/synthetic_data_669/synthetic_Context-Inconsistency_llama-3-70b-instruct_dev.json \
        --output_dir ../models/prm_llama3_Context-Inconsistency_6/ \
        --do_train \
        --do_eval \
        --do_predict \
        --bf16 \
        --num_train_epochs 50 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --eval_strategy epoch \
        --logging_strategy epoch \
        --save_strategy epoch \
        --load_best_model_at_end \
        --metric_for_best_model overall_accuracy \
        --max_seq_length 1536 \
        --report_to wandb \
        --save_total_limit 2 \
        --learning_rate 0.000005 \
        --weight_decay 0.001 \
        --warmup_ratio 0.1

# train reward model for PRM
torchrun \
    --nproc_per_node $gpu_num \
    --nnodes=1 \ 
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    ./reward_modeling/run_fg_rm_llama3.py \
        --model_name_or_path /data/models/huggingface-format/llama-3-8b/ \
        --train_file ../data/hallucination_sample/synthetic_data_669/synthetic_Instruction-Inconsistency_llama-3-70b-instruct_train.json \
        --validation_file ../data/hallucination_sample/synthetic_data_669/synthetic_Instruction-Inconsistency_llama-3-70b-instruct_dev.json \
        --test_file ../data/hallucination_sample/synthetic_data_669/synthetic_Instruction-Inconsistency_llama-3-70b-instruct_dev.json \
        --output_dir ../models/prm_llama3_Instruction-Inconsistency_6/ \
        --do_train \
        --do_eval \
        --do_predict \
        --bf16 \
        --num_train_epochs 50 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --eval_strategy epoch \
        --logging_strategy epoch \
        --save_strategy epoch \
        --load_best_model_at_end \
        --metric_for_best_model overall_accuracy \
        --max_seq_length 1536 \
        --report_to wandb \
        --save_total_limit 2 \
        --learning_rate 0.000005 \
        --weight_decay 0.001 \
        --warmup_ratio 0.1

# train reward model for PRM
torchrun \
    --nproc_per_node $gpu_num \
    --nnodes=1 \ 
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    ./reward_modeling/run_fg_rm_llama3.py \
        --model_name_or_path /data/models/huggingface-format/llama-3-8b/ \
        --train_file ../data/hallucination_sample/synthetic_data_669/synthetic_Fabrication_llama-3-70b-instruct_train.json \
        --validation_file ../data/hallucination_sample/synthetic_data_669/synthetic_Fabrication_llama-3-70b-instruct_dev.json \
        --test_file ../data/hallucination_sample/synthetic_data_669/synthetic_Fabrication_llama-3-70b-instruct_dev.json \
        --output_dir ../models/prm_llama3_Fabrication_6/ \
        --do_train \
        --do_eval \
        --do_predict \
        --bf16 \
        --num_train_epochs 50 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --eval_strategy epoch \
        --logging_strategy epoch \
        --save_strategy epoch \
        --load_best_model_at_end \
        --metric_for_best_model overall_accuracy \
        --max_seq_length 1536 \
        --report_to wandb \
        --save_total_limit 2 \
        --learning_rate 0.000005 \
        --weight_decay 0.001 \
        --warmup_ratio 0.1

