#!/bin/bash

# 设置基本参数
MODEL_NAME="deepseek-ai/deepseek-coder-1.3b-base"  # 使用较小的模型
OUTPUT_DIR="./output/deepseek_lora_fast"           # 输出目录
DATA_DIR="./data"                                  # 数据目录
PROCESSED_DATA_DIR="./processed_data"         # 处理后的数据目录
LOG_FILE="output_fast.txt"                         # 日志文件

# 数据文件
TRAIN_FILE="${DATA_DIR}/cur_train.jsonl"
DEV_FILE="${DATA_DIR}/cur_dev.jsonl"
TEST_FILE="${DATA_DIR}/new_train.jsonl"

# 训练参数 - 优化速度
BATCH_SIZE=64                 # 增大批次大小
GRAD_ACCUM_STEPS=2            # 最小梯度累积步数
LEARNING_RATE=5e-4            # 略微提高学习率加速收敛
NUM_EPOCHS=1                  # 减少训练轮数
MAX_LENGTH=512                # 减少序列长度
MAX_SOURCE_LENGTH=$MAX_LENGTH # 最大源序列长度
MAX_TARGET_LENGTH=$MAX_LENGTH # 最大目标序列长度
BEAM_SIZE=1                   # 评估时减少束搜索大小

# LoRA参数 - 减少参数量
LORA_R=4                      # 减小LoRA秩
LORA_ALPHA=8                  # 减小LoRA alpha
LORA_DROPOUT=0.05             # 保持dropout不变

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $PROCESSED_DATA_DIR

# 选择操作模式
MODE=${1:-"train_fast"}  # 默认为快速训练模式

# 记录开始时间和命令
echo "====================================="
echo "开始时间: $(date)"
echo "运行模式: $MODE"
echo "====================================="

# 环境变量设置
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export FORCE_COLOR=1
export CUDA_LAUNCH_BLOCKING=0  # 禁用CUDA同步以提高速度

# 创建DeepSpeed配置
cat > ds_config.json << 'EOL'
{
  "train_batch_size": 16,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 5e-4,
      "weight_decay": 0.0
    }
  },
  "fp16": {
    "enabled": true,
    "auto_cast": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  },
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true
  }
}
EOL

case $MODE in
  "train_fast")
    # 快速训练模式
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
      --processed_data_dir $PROCESSED_DATA_DIR \
      --max_train_samples 5000 \
      --eval_steps 500 \
      --save_steps 500 \
      --fp16
    ;;
    
  "train_deepspeed")
    # DeepSpeed训练模式 (如果有多GPU)
    deepspeed train_deepseek_lora.py \
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
      --processed_data_dir $PROCESSED_DATA_DIR \
      --max_train_samples 5000 \
      --eval_steps 500 \
      --save_steps 500 \
      --deepspeed ds_config.json
    ;;
    
  *)
    echo "未知的模式: $MODE"
    echo "可用模式: train_fast, train_deepspeed"
    exit 1
    ;;
esac

# 记录结束时间
echo "====================================="
echo "结束时间: $(date)"
echo "=====================================" 