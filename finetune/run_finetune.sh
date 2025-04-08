#!/bin/bash

# 设置环境变量
# export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用的GPU设备
# export PYTHONPATH="$PYTHONPATH:$(pwd)"

# 训练数据路径
DATA_PATH="data/train.jsonl"

# 测试数据路径
TEST_DATA_PATH="data/test.jsonl"

# 模型参数
MODEL_NAME="deepseek-ai/deepseek-coder-1.3b-instruct"
OUTPUT_DIR="./output/deepseek-lora"

# LoRA参数
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# 训练参数
BATCH_SIZE=8
GRAD_ACCUM=4
LR=2e-5
EPOCHS=3
MAX_LENGTH=512
WARMUP_RATIO=0.03

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 创建日志目录
LOG_DIR="./logs"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

# 使用tee命令将输出同时显示在终端和写入到日志文件
python finetune_deepseek.py \
    --model_name_or_path $MODEL_NAME \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --use_lora True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --target_modules $TARGET_MODULES \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --warmup_ratio $WARMUP_RATIO \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --model_max_length $MAX_LENGTH \
    --fp16 True \
    --report_to "tensorboard" \
    --ddp_find_unused_parameters False 2>&1 | tee $LOG_FILE

echo "操作完成！模型保存在 $OUTPUT_DIR"
echo "训练日志保存在 $LOG_FILE"