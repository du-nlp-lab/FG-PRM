set -e

gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

# train reward model for ORM
torchrun --nproc_per_node $gpu_num --standalone --nnodes=1 ./reward_modeling/run_fg_rm.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ../data/hallucination_sample/synthetic_Calculation-Error_longformer-base-4096_orm_train.json \
                --validation_file ../data/hallucination_sample/synthetic_Calculation-Error_longformer-base-4096_orm_dev.json \
                --test_file ../data/hallucination_sample/synthetic_Calculation-Error_longformer-base-4096_orm_dev.json \
                --output_dir ../models/orm_longformer_Calculation-Error \
                --do_train \
                --do_eval \
                --do_predict \
                --bf16 \
                --num_train_epochs 50 \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 16 \
                --evaluation_strategy epoch \
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

torchrun --nproc_per_node $gpu_num --standalone --nnodes=1 ./reward_modeling/run_fg_rm.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ../data/hallucination_sample/synthetic_Context-Inconsistency_longformer-base-4096_orm_train.json \
                --validation_file ../data/hallucination_sample/synthetic_Context-Inconsistency_longformer-base-4096_orm_dev.json \
                --test_file ../data/hallucination_sample/synthetic_Context-Inconsistency_longformer-base-4096_orm_dev.json \
                --output_dir ../models/orm_longformer_Context-Inconsistency \
                --do_train \
                --do_eval \
                --do_predict \
                --bf16 \
                --num_train_epochs 50 \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 16 \
                --evaluation_strategy epoch \
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

torchrun --nproc_per_node $gpu_num --standalone --nnodes=1 ./reward_modeling/run_fg_rm.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ../data/hallucination_sample/synthetic_Fabrication_longformer-base-4096_orm_train.json \
                --validation_file ../data/hallucination_sample/synthetic_Fabrication_longformer-base-4096_orm_dev.json \
                --test_file ../data/hallucination_sample/synthetic_Fabrication_longformer-base-4096_orm_dev.json \
                --output_dir ../models/orm_longformer_Fabrication \
                --do_train \
                --do_eval \
                --do_predict \
                --bf16 \
                --num_train_epochs 50 \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 16 \
                --evaluation_strategy epoch \
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

torchrun --nproc_per_node $gpu_num --standalone --nnodes=1 ./reward_modeling/run_fg_rm.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ../data/hallucination_sample/synthetic_Factual-Inconsistency_longformer-base-4096_orm_train.json \
                --validation_file ../data/hallucination_sample/synthetic_Factual-Inconsistency_longformer-base-4096_orm_dev.json \
                --test_file ../data/hallucination_sample/synthetic_Factual-Inconsistency_longformer-base-4096_orm_dev.json \
                --output_dir ../models/orm_longformer_Factual-Inconsistency \
                --do_train \
                --do_eval \
                --do_predict \
                --bf16 \
                --num_train_epochs 50 \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 16 \
                --evaluation_strategy epoch \
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

torchrun --nproc_per_node $gpu_num --standalone --nnodes=1 ./reward_modeling/run_fg_rm.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ../data/hallucination_sample/synthetic_Instruction-Inconsistency_longformer-base-4096_orm_train.json \
                --validation_file ../data/hallucination_sample/synthetic_Instruction-Inconsistency_longformer-base-4096_orm_dev.json \
                --test_file ../data/hallucination_sample/synthetic_Instruction-Inconsistency_longformer-base-4096_orm_dev.json \
                --output_dir ../models/orm_longformer_Instruction-Inconsistency \
                --do_train \
                --do_eval \
                --do_predict \
                --bf16 \
                --num_train_epochs 50 \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 16 \
                --evaluation_strategy epoch \
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

torchrun --nproc_per_node $gpu_num --standalone --nnodes=1 ./reward_modeling/run_fg_rm.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ../data/hallucination_sample/synthetic_Logical-Inconsistency_longformer-base-4096_orm_train.json \
                --validation_file ../data/hallucination_sample/synthetic_Logical-Inconsistency_longformer-base-4096_orm_dev.json \
                --test_file ../data/hallucination_sample/synthetic_Logical-Inconsistency_longformer-base-4096_orm_dev.json \
                --output_dir ../models/orm_longformer_Logical-Inconsistency \
                --do_train \
                --do_eval \
                --do_predict \
                --bf16 \
                --num_train_epochs 50 \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 16 \
                --evaluation_strategy epoch \
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


