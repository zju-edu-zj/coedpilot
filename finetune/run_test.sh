#!/bin/bash

# 设置环境变量
# export CUDA_VISIBLE_DEVICES=0  # 使用的GPU设备

# 基础模型路径
BASE_MODEL="deepseek-ai/deepseek-coder-1.3b-instruct"

# LoRA模型路径
LORA_MODEL="./output/deepseek-lora"

# 测试数据路径
TEST_DATA="./data/test.jsonl"

# 输出目录和文件
OUTPUT_DIR="./test_results"
OUTPUT_FILE="${OUTPUT_DIR}/test_results.json"
BLEU_OUTPUT="test_pred_gold.json"

# 测试参数
MAX_SAMPLES=100  # 设置为None或删除此行以测试所有样本
TEMPERATURE=0.2
MAX_NEW_TOKENS=512
BATCH_SIZE=4     # 批处理大小
BEAM_SIZE=5      # 束搜索大小

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行测试脚本
python test_model.py \
    --base_model $BASE_MODEL \
    # --lora_model $LORA_MODEL \
    --test_data $TEST_DATA \
    --output_file $OUTPUT_FILE \
    --bleu_output $BLEU_OUTPUT \
    --max_samples $MAX_SAMPLES \
    --temperature $TEMPERATURE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --batch_size $BATCH_SIZE \
    --beam_size $BEAM_SIZE

echo "测试完成！结果保存在 $OUTPUT_FILE"
echo "BLEU评估结果保存在 ${OUTPUT_DIR}/${BLEU_OUTPUT}" 