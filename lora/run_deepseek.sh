#!/bin/bash

# 设置基本参数
MODEL_NAME="deepseek-ai/deepseek-coder-1.3b-base"  # 模型名称
OUTPUT_DIR="./output/deepseek_lora"                # 输出目录
DATA_DIR="./data"                                  # 数据目录
PROCESSED_DATA_DIR="./processed_data"              # 处理后的数据目录
LOG_FILE="output.txt"                              # 日志文件

# 数据文件
TRAIN_FILE="${DATA_DIR}/cur_train.jsonl"
DEV_FILE="${DATA_DIR}/cur_dev.jsonl"
TEST_FILE="${DATA_DIR}/new_train.jsonl"

# 训练参数
BATCH_SIZE=4                  # 批次大小
GRAD_ACCUM_STEPS=4            # 梯度累积步数
LEARNING_RATE=2e-4            # 学习率
NUM_EPOCHS=3                  # 训练轮数
MAX_SOURCE_LENGTH=1024        # 最大源序列长度
MAX_TARGET_LENGTH=256         # 最大目标序列长度
BEAM_SIZE=5                   # 束搜索大小

# LoRA参数
LORA_R=8                      # LoRA秩
LORA_ALPHA=16                 # LoRA alpha
LORA_DROPOUT=0.05             # LoRA dropout

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $PROCESSED_DATA_DIR

# 选择操作模式
MODE=${1:-"train"}  # 默认为训练模式

# 记录开始时间和命令
echo "====================================="
echo "开始时间: $(date)"
echo "运行模式: $MODE"
echo "====================================="

# 设置PYTHONIOENCODING以确保正确处理Unicode
export PYTHONIOENCODING=utf-8

# 设置PYTHONUNBUFFERED以确保输出不被缓冲
export PYTHONUNBUFFERED=1

# 设置FORCE_COLOR环境变量，强制tqdm显示颜色
export FORCE_COLOR=1

case $MODE in
  "process_data")
    # 只处理数据并保存
    python train_deepseek_lora.py \
      --model_name_or_path $MODEL_NAME \
      --output_dir $OUTPUT_DIR \
      --train_filename $TRAIN_FILE \
      --dev_filename $DEV_FILE \
      --test_filename $TEST_FILE \
      --save_processed_data \
      --processed_data_output_dir $PROCESSED_DATA_DIR \
      --max_source_length $MAX_SOURCE_LENGTH \
      --max_target_length $MAX_TARGET_LENGTH
    ;;
    
  "train")
    # 训练模式
    python train_deepseek_lora.py \
      --model_name_or_path $MODEL_NAME \
      --output_dir $OUTPUT_DIR \
      --train_filename $TRAIN_FILE \
      --dev_filename $DEV_FILE \
      --max_source_length $MAX_SOURCE_LENGTH \
      --max_target_length $MAX_TARGET_LENGTH \
      --train_batch_size $BATCH_SIZE \
      --eval_batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
      --learning_rate $LEARNING_RATE \
      --num_train_epochs $NUM_EPOCHS \
      --beam_size $BEAM_SIZE \
      --lora_r $LORA_R \
      --lora_alpha $LORA_ALPHA \
      --lora_dropout $LORA_DROPOUT \
      --use_8bit \
      --do_train \
      --do_eval \
      --processed_data_dir $PROCESSED_DATA_DIR
    ;;
    
  "test")
    # 测试模式
    python train_deepseek_lora.py \
      --model_name_or_path $MODEL_NAME \
      --output_dir $OUTPUT_DIR \
      --test_filename $TEST_FILE \
      --max_source_length $MAX_SOURCE_LENGTH \
      --max_target_length $MAX_TARGET_LENGTH \
      --eval_batch_size $BATCH_SIZE \
      --beam_size $BEAM_SIZE \
      --do_test \
      --processed_data_dir $PROCESSED_DATA_DIR
    ;;
    
  "train_4bit")
    # 使用4位量化训练
    python train_deepseek_lora.py \
      --model_name_or_path $MODEL_NAME \
      --output_dir $OUTPUT_DIR \
      --train_filename $TRAIN_FILE \
      --dev_filename $DEV_FILE \
      --max_source_length $MAX_SOURCE_LENGTH \
      --max_target_length $MAX_TARGET_LENGTH \
      --train_batch_size $BATCH_SIZE \
      --eval_batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
      --learning_rate $LEARNING_RATE \
      --num_train_epochs $NUM_EPOCHS \
      --beam_size $BEAM_SIZE \
      --lora_r $LORA_R \
      --lora_alpha $LORA_ALPHA \
      --lora_dropout $LORA_DROPOUT \
      --use_4bit \
      --do_train \
      --do_eval \
      --processed_data_dir $PROCESSED_DATA_DIR
    ;;
    
  "direct_test")
    # 直接测试模式，无需预处理数据
    python train_deepseek_lora.py \
      --model_name_or_path $MODEL_NAME \
      --output_dir $OUTPUT_DIR \
      --test_filename $TEST_FILE \
      --max_source_length $MAX_SOURCE_LENGTH \
      --max_target_length $MAX_TARGET_LENGTH \
      --eval_batch_size $BATCH_SIZE \
      --beam_size $BEAM_SIZE \
      --do_test
    ;;
    
  *)
    echo "未知的模式: $MODE"
    echo "可用模式: process_data, train, test, train_4bit, direct_test"
    exit 1
    ;;
esac

# 记录结束时间
echo "====================================="
echo "结束时间: $(date)"
echo "=====================================" 