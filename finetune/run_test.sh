#!/bin/bash

# 设置环境变量
# export CUDA_VISIBLE_DEVICES=0  # 使用的GPU设备

# 基础模型路径
BASE_MODEL="deepseek-ai/deepseek-coder-1.3b-instruct"

# LoRA模型路径
LORA_MODEL="./output/deepseek-lora"

# 测试数据路径
TEST_DATA="./data/test.jsonl"

# 输出文件
OUTPUT_FILE="./test_results.json"

# 测试参数
MAX_SAMPLES=100  # 设置为None或删除此行以测试所有样本
TEMPERATURE=0.2
MAX_NEW_TOKENS=512

# 运行测试脚本
python finetune/test_model.py \
    --base_model $BASE_MODEL \
    --lora_model $LORA_MODEL \
    --test_data $TEST_DATA \
    --output_file $OUTPUT_FILE \
    --max_samples $MAX_SAMPLES \
    --temperature $TEMPERATURE \
    --max_new_tokens $MAX_NEW_TOKENS

echo "测试完成！结果保存在 $OUTPUT_FILE" 