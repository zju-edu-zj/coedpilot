#!/bin/bash

# 设置环境变量
# export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用的GPU设备
# export PYTHONPATH="$PYTHONPATH:$(pwd)"

# 训练数据路径
DATA_PATH="./data/train.json"

# 测试数据路径
TEST_DATA_PATH="./data/test.json"

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

# 操作模式：train, test, 或 both
MODE=${1:-both}

# 运行训练和/或测试脚本
if [ "$MODE" == "train" ] || [ "$MODE" == "both" ]; then
    python lora/finetune_deepseek.py \
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
        --ddp_find_unused_parameters False
fi

if [ "$MODE" == "test" ] || [ "$MODE" == "both" ]; then
    python lora/finetune_deepseek.py \
        --model_name_or_path $MODEL_NAME \
        --test_data_path $TEST_DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --do_eval True \
        --model_max_length $MAX_LENGTH \
        --fp16 True
fi

echo "操作完成！模型保存在 $OUTPUT_DIR"