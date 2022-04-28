#! /bin/bash
# Fine-tune FactCC model

# UPDATE PATHS BEFORE RUNNING SCRIPT
export CODE_PATH=/home/gaomq/factCC/modeling # absolute path to modeling directory
export DATA_PATH=/home/gaomq/factCC/samsum/data4train # absolute path to data directory
export OUTPUT_PATH=/home/gaomq/factCC/samsum/checkpoint_btrans # absolute path to model checkpoint

export TASK_NAME=factcc_generated
export MODEL_NAME=bert-base-uncased
export CKPT_PATH=/home/gaomq/factCC/bert-pretrained-checkpoint
<< EOF
python3 $CODE_PATH/run.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --do_lower_case \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 12 \
  --learning_rate 2e-5 \
  --num_train_epochs 10.0 \
  --data_dir $DATA_PATH \
  --model_type bert \
  --model_name_or_path $MODEL_NAME \
  --output_dir $OUTPUT_PATH/$MODEL_NAME-$TASK_NAME-finetune-$RANDOM/
EOF

CUDA_VISIBLE_DEVICES=0 python3 $CODE_PATH/run.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --do_lower_case \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 6 \
  --learning_rate 2e-5 \
  --num_train_epochs 10.0 \
  --data_dir $DATA_PATH \
  --model_type bert \
  --tokenizer_name $MODEL_NAME \
  --config_name $MODEL_NAME \
  --model_name_or_path $CKPT_PATH \
  --output_dir $OUTPUT_PATH/$MODEL_NAME-$TASK_NAME-finetune/