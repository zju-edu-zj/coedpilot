# coding=utf-8
from __future__ import absolute_import
import os
import sys
import json
import random
import logging
import argparse
from io import open
import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
    RandomSampler,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from tqdm import tqdm, trange
import bleu
import warnings
import transformers.generation.utils
from torch.utils.tensorboard import SummaryWriter

# 添加方法二的代码在这里
import transformers.generation.utils

# 直接修补generate方法中的警告检查
original_generate = transformers.generation.utils.GenerationMixin.generate

def patched_generate(self, *args, **kwargs):
    # 保存原始的logger.warning函数
    original_warning = transformers.generation.utils.logger.warning
    
    # 创建一个过滤特定警告的函数
    def filtered_warning(message, *args, **kwargs):
        if "right-padding was detected" not in message:
            original_warning(message, *args, **kwargs)
    
    # 替换warning函数
    transformers.generation.utils.logger.warning = filtered_warning
    
    # 调用原始generate函数
    result = original_generate(self, *args, **kwargs)
    
    # 恢复原始warning函数
    transformers.generation.utils.logger.warning = original_warning
    
    return result

# 替换原始函数
transformers.generation.utils.GenerationMixin.generate = patched_generate

# 使用部分匹配来过滤关于padding_side的警告
warnings.filterwarnings("ignore", message=".*padding was detected.*")

# 导入数据处理功能
from data_processor import (
    read_examples,
    convert_examples_to_features,
    prepare_training_features,
    prepare_validation_features,
    prepare_test_features,
    save_features,
    load_features,
    process_and_save_data,
)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""

    def __init__(self, idx, source, target, edit_ops):
        self.idx = idx
        self.source = source
        self.target = target
        self.edit_ops = edit_ops


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        '--model_name_or_path',
        default=None,
        type=str,
        required=True,
        help='Path to pre-trained model: e.g. deepseek-ai/deepseek-coder-6.7b-base',
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        type=str,
        required=True,
        help='The output directory where the model predictions and checkpoints will be written.',
    )
    parser.add_argument(
        '--train_filename',
        default=None,
        type=str,
        help='The train filename. Should contain the .jsonl files for this task.',
    )
    parser.add_argument(
        '--dev_filename',
        default=None,
        type=str,
        help='The dev filename. Should contain the .jsonl files for this task.',
    )
    parser.add_argument(
        '--test_filename',
        default=None,
        type=str,
        help='The test filename. Should contain the .jsonl files for this task.',
    )
    parser.add_argument(
        '--max_source_length',
        default=512,
        type=int,
        help='The maximum total source sequence length after tokenization.',
    )
    parser.add_argument(
        '--max_target_length',
        default=128,
        type=int,
        help='The maximum total target sequence length after tokenization.',
    )
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true', help='Whether to run eval on the dev set.')
    parser.add_argument('--do_test', action='store_true', help='Whether to run eval on the test set.')
    parser.add_argument('--no_cuda', action='store_true', help='Avoid using CUDA when available')
    parser.add_argument(
        '--train_batch_size',
        default=8,
        type=int,
        help='Batch size per GPU/CPU for training.',
    )
    parser.add_argument(
        '--eval_batch_size',
        default=8,
        type=int,
        help='Batch size per GPU/CPU for evaluation.',
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of updates steps to accumulate before performing a backward/update pass.',
    )
    parser.add_argument(
        '--learning_rate',
        default=2e-4,
        type=float,
        help='The initial learning rate for Adam.',
    )
    parser.add_argument(
        '--beam_size', default=5, type=int, help='beam size for beam search'
    )
    parser.add_argument(
        '--weight_decay', default=0.0, type=float, help='Weight decay if we apply some.',
    )
    parser.add_argument(
        '--adam_epsilon', default=1e-8, type=float, help='Epsilon for Adam optimizer.',
    )
    parser.add_argument(
        '--max_grad_norm', default=1.0, type=float, help='Max gradient norm.'
    )
    parser.add_argument(
        '--num_train_epochs',
        default=3,
        type=int,
        help='Total number of training epochs to perform.',
    )
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    
    # LoRA specific arguments
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--use_8bit', action='store_true', help='Use 8-bit quantization')
    parser.add_argument('--use_4bit', action='store_true', help='Use 4-bit quantization')
    
    # New arguments for processed data
    parser.add_argument(
        '--processed_data_dir',
        default=None,
        type=str,
        help='Directory containing processed features. If provided, will load features from this directory instead of processing raw data.',
    )
    parser.add_argument(
        '--save_processed_data',
        action='store_true',
        help='Whether to save processed features to disk for future use.',
    )
    parser.add_argument(
        '--processed_data_output_dir',
        default='processed_data',
        type=str,
        help='Directory to save processed features if --save_processed_data is set.',
    )
    
    parser.add_argument('--max_train_samples', type=int, default=None, help='训练时使用的最大样本数量')
    parser.add_argument('--max_eval_samples', type=int, default=None, help='评估时使用的最大样本数量')
    parser.add_argument('--eval_every_n_epochs', type=int, default=1, help='每隔多少个epoch评估一次')
    
    # New parameters for testing
    parser.add_argument('--max_test_samples', type=int, default=None, help='测试时使用的最大样本数量')
    parser.add_argument('--test_batch_size', type=int, default=None, help='测试时使用的批量大小，默认与eval_batch_size相同')
    
    # 添加TensorBoard相关参数
    parser.add_argument(
        '--tensorboard_dir',
        default=None,
        type=str,
        help='TensorBoard日志目录，默认为output_dir/runs',
    )
    parser.add_argument(
        '--log_steps',
        default=10,
        type=int,
        help='每多少步记录一次训练损失到TensorBoard',
    )
    
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Set seed
    set_seed(args.seed)
    
    # Make output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 设置TensorBoard日志目录
    if args.tensorboard_dir is None:
        # 添加时间戳以区分不同运行
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.tensorboard_dir = os.path.join(args.output_dir, f'runs_{timestamp}')
    else:
        # 如果用户指定了目录，可以选择是否添加时间戳
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.tensorboard_dir = os.path.join(args.tensorboard_dir, timestamp)

    # 确保目录存在
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # 设置左侧填充
    # logger.info(f"tokenizer padding side: {tokenizer.padding_side}")
    # tokenizer.padding_side = 'left'
    # logger.info(f"current tokenizer padding side: {tokenizer.padding_side}")
    special_tokens = {
        "additional_special_tokens": [
            "<ADD_CODE>", 
            "<REPLACE_CODE>", 
            "<KEEP_CODE>",
            "<ADD>", 
            "<REMOVE>"
        ]
    }
    logger.info(f"len of tokenizer: {len(tokenizer)}, vocab size: {tokenizer.vocab_size}")
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    logger.info(f"Added {num_added_tokens} new special tokens, current vocab size: {tokenizer.vocab_size}, current len of tokenizer: {len(tokenizer)}")
    
    # 设置量化参数
    compute_dtype = torch.float16
    if args.use_8bit:
        load_in_8bit = True
        load_in_4bit = False
    elif args.use_4bit:
        load_in_8bit = False
        load_in_4bit = True
    else:
        load_in_8bit = False
        load_in_4bit = False
    
     # 添加一个专门用于处理和保存数据的选项
    if args.save_processed_data and not (args.do_train or args.do_eval or args.do_test):
        logger.info("Processing and saving all datasets...")
        process_and_save_data(
            args.train_filename, 
            args.dev_filename, 
            args.test_filename, 
            tokenizer, 
            args, 
            args.processed_data_output_dir
        )
        logger.info("All datasets processed and saved.")
        return
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        device_map="auto",
        use_cache=False  # 禁用KV缓存以兼容梯度检查点
    )
    
    # 为LoRA准备模型
    if load_in_8bit or load_in_4bit:
        logger.info(f"Preparing model for kbit training: 8bit={load_in_8bit}, 4bit={load_in_4bit}")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # 配置LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # 应用LoRA配置
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # 打印可训练参数比例
    
    # 调整嵌入大小以适应新添加的特殊标记
    # 打印模型的embedding层大小和tokenizer大小
    logger.info(f"模型embedding层大小: {model.get_input_embeddings().weight.shape}")
    logger.info(f"tokenizer词表大小: {len(tokenizer)}")
    
    # 修正：确保嵌入层大小不会被缩小，只会扩大
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        # 只有当tokenizer大于模型嵌入层时才调整大小
        model.resize_token_embeddings(len(tokenizer))
    else:
        # 如果tokenizer小于模型嵌入层，则将tokenizer扩展到模型嵌入层大小
        logger.info(f"分词器大小小于模型嵌入层大小，保持模型嵌入层不变")
        # 可能需要确保tokenizer能够处理所有模型支持的token
    
    # 调整后再次打印embedding层大小
    logger.info(f"调整后的embedding层大小: {model.get_input_embeddings().weight.shape}")
    
    if args.do_train:
        # 初始化TensorBoard
        tb_writer = SummaryWriter(args.tensorboard_dir)
        logger.info(f"TensorBoard日志将保存到: {args.tensorboard_dir}")
        
        if args.processed_data_dir:
            # 从处理好的文件加载特征
            train_features = load_features(os.path.join(args.processed_data_dir, 'train_features.pt'))
            logger.info(f"Loaded {len(train_features)} training features")
        else:
            # 从原始文件处理特征
            logger.info("unexpected, please save processed data first")
            exit()
        
        # 在数据加载部分添加数据采样
        if args.max_train_samples is not None and len(train_features) > args.max_train_samples:
            logger.info(f"Using {args.max_train_samples} samples out of {len(train_features)} training samples")
            # 随机采样以保持数据分布
            random.seed(42)
            train_features = random.sample(train_features, args.max_train_samples)
        
        all_input_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_labels = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        
        train_data = TensorDataset(all_input_ids, all_attention_mask, all_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=args.train_batch_size // args.gradient_accumulation_steps,
        )
        
        # 准备优化器和学习率调度器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay
        )
        
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total
        )
        
        # 开始训练
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epochs = %d", args.num_train_epochs)
        
        model.train()
        global_step = 0
        tr_loss = 0.0
        best_bleu = 0
        
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch}")
            epoch_loss = 0
            
            for step, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, labels = batch
                
                # 检查输入和标签的形状
                if input_ids.shape != labels.shape:
                    logger.warning(f"Input IDs shape: {input_ids.shape}, Labels shape: {labels.shape}")
                    raise ValueError("输入和标签形状不匹配，请检查数据处理流程")
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    
                loss.backward()
                epoch_loss += loss.item()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # 记录训练损失到TensorBoard
                    if global_step % args.log_steps == 0:
                        tb_writer.add_scalar('train/loss', loss.item() * args.gradient_accumulation_steps, global_step)
                        tb_writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
                
                bar.set_postfix(loss=epoch_loss/(step+1))
            
            # 记录每个epoch的平均损失
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            tb_writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
            tr_loss += epoch_loss
            
            # 保存每个epoch的模型
            output_dir = os.path.join(args.output_dir, f'checkpoint-epoch-{epoch}')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # 评估模型
            if args.do_eval and (epoch + 1) % args.eval_every_n_epochs == 0:
                # 保存当前模型状态
                model_was_training = model.training
                
                # 启用缓存以加速生成
                model.config.use_cache = True
                
                # 设置为评估模式
                model.eval()
                
                if args.processed_data_dir:
                    eval_features = load_features(os.path.join(args.processed_data_dir, 'dev_features.pt'))
                    # 需要加载原始示例用于评估
                    eval_examples = read_examples(args.dev_filename)
                else:
                    logger.info("unexpected, please save processed data first")
                    exit()
                
                # 限制评估样本数量以加速验证
                max_eval_samples = min(len(eval_features), args.max_eval_samples if args.max_eval_samples else len(eval_features))
                if max_eval_samples < len(eval_features):
                    logger.info(f"使用 {max_eval_samples} 个样本进行评估 (共 {len(eval_features)} 个)")
                    # 随机采样或使用前N个样本
                    eval_features = eval_features[:max_eval_samples]
                    eval_examples = eval_examples[:max_eval_samples]
                
                all_input_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                all_attention_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_input_ids, all_attention_mask)
                
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(
                    eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
                )
                
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_features))
                logger.info("  Batch size = %d", args.eval_batch_size)
                
                predictions = []
                
                # 确保pad_token_id已设置
                if model.config.pad_token_id is None:
                    logger.info(f"Setting pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
                    model.config.pad_token_id = tokenizer.eos_token_id

                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, attention_mask = batch
                    
                    with torch.no_grad():
                        generated_ids = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=args.max_target_length,
                            num_beams=args.beam_size,
                            num_return_sequences=1,
                            do_sample=False,
                            early_stopping=True,
                            pad_token_id=model.config.pad_token_id,
                            # 添加以下参数以加速生成
                            repetition_penalty=1.0,
                            length_penalty=1.0,
                            no_repeat_ngram_size=0,
                        )
                    
                    for i, g_ids in enumerate(generated_ids):
                        # 找到输入结束的位置
                        input_length = len(input_ids[i])
                        gen_text = tokenizer.decode(g_ids[input_length:], skip_special_tokens=True)
                        predictions.append(gen_text)
                
                # 恢复原始状态
                model.config.use_cache = False
                if model_was_training:
                    model.train()
                
                # 计算BLEU分数
                references = [ex.target for ex in eval_examples]
                bleu_output_dict = {}
                for idx, (pred, ex) in enumerate(zip(predictions, eval_examples)):
                    bleu_output_dict[str(ex.idx)] = ([pred], ex.target)
                
                with open(os.path.join(args.output_dir, f'eval_epoch_{epoch}_pred_gold.json'), 'w') as f:
                    json.dump(bleu_output_dict, f, indent=2)
                
                # 使用bleu模块的computeMaps_multiple和bleuFromMaps方法计算BLEU分数
                (goldMap, predictionMap) = bleu.computeMaps_multiple(
                    os.path.join(args.output_dir, f'eval_epoch_{epoch}_pred_gold.json'),
                    1  # 因为eval只有一个预测，所以beam_size=1
                )
                bleu_score = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
                
                # 记录BLEU分数到TensorBoard
                tb_writer.add_scalar('eval/bleu', bleu_score, epoch)
                logger.info(f"Evaluate BLEU score at epoch {epoch}: {bleu_score}")
                
                # 记录一些生成示例到TensorBoard
                num_examples = min(5, len(predictions))
                for i in range(num_examples):
                    tb_writer.add_text(
                        f'eval/example_{i}',
                        f"**Source**: {tokenizer.decode(eval_features[i].source_ids, skip_special_tokens=True)}\n\n"
                        f"**Prediction**: {predictions[i]}\n\n"
                        f"**Target**: {eval_examples[i].target}",
                        epoch
                    )
                
                # 保存最佳模型
                if bleu_score > best_bleu:
                    best_bleu = bleu_score
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
        
        # 保存最终模型
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # 关闭TensorBoard写入器
        tb_writer.close()
    
    # 测试模型
    if args.do_test:
        # 初始化TensorBoard (如果尚未初始化)
        if not args.do_train:
            tb_writer = SummaryWriter(args.tensorboard_dir)
            logger.info(f"TensorBoard日志将保存到: {args.tensorboard_dir}")
        
        if args.processed_data_dir:
            test_features = load_features(os.path.join(args.processed_data_dir, 'test_features.pt'))
            test_examples = read_examples(args.test_filename)
        else:
            # 直接处理测试数据
            logger.info("处理测试数据...")
            test_examples = read_examples(args.test_filename)
            test_features, prompts = convert_examples_to_features(
                examples=test_examples,
                tokenizer=tokenizer,
                args=args
            )
            logger.info(f"处理了 {len(test_features)} 条测试特征，prompt: {prompts}")
        
        # 限制测试样本数量
        max_test_samples = min(len(test_features), args.max_test_samples if args.max_test_samples else len(test_features))
        if max_test_samples < len(test_features):
            logger.info(f"使用 {max_test_samples} 个样本进行测试 (共 {len(test_features)} 个)")
            # 随机采样或使用前N个样本
            test_features = test_features[:max_test_samples]
            test_examples = test_examples[:max_test_samples]
        
        # 加载最佳模型
        best_model_path = os.path.join(args.output_dir, 'checkpoint-best-bleu')
        if os.path.exists(best_model_path):
            tokenizer = AutoTokenizer.from_pretrained(best_model_path, trust_remote_code=True)
            logger.info(f"len of tokenizer: {len(tokenizer)}, vocab size: {tokenizer.vocab_size}")
            model = AutoModelForCausalLM.from_pretrained(
                best_model_path,
                trust_remote_code=True,
                torch_dtype=compute_dtype,
                device_map="auto",
                use_cache=True  # 在推理时启用KV缓存以加速生成
            )
        else:
            logger.info("找不到最佳模型，使用预训练模型")
            model.config.use_cache = True
            
        # 使用更大的批量大小进行测试
        test_batch_size = args.test_batch_size if args.test_batch_size else args.eval_batch_size
        
        all_input_ids = torch.tensor([f.source_ids for f in test_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.source_mask for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_attention_mask)
        
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=test_batch_size
        )
        
        logger.info("***** Running testing *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", test_batch_size)
        
        model.eval()
        predictions = []
        
        # 确保pad_token_id已设置
        if model.config.pad_token_id is None:
            logger.info(f"Setting pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
            model.config.pad_token_id = tokenizer.eos_token_id

        for batch in tqdm(test_dataloader, desc="Testing"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask = batch
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_target_length,
                    num_beams=args.beam_size,
                    num_return_sequences=args.beam_size,
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=model.config.pad_token_id,
                    # 添加以下参数以加速生成
                    repetition_penalty=1.0,
                    length_penalty=1.0,
                    no_repeat_ngram_size=0,
                )
            
            for i, beam_outputs in enumerate(generated_ids.reshape(input_ids.shape[0], args.beam_size, -1)):
                beam_results = []
                for beam_output in beam_outputs:
                    # 找到输入结束的位置
                    input_length = len(input_ids[i])
                    gen_text = tokenizer.decode(beam_output[input_length:], skip_special_tokens=True)
                    beam_results.append(gen_text)
                predictions.append(beam_results)
        
        # 计算BLEU分数
        references = [ex.target for ex in test_examples]
        best_predictions = [p[0] for p in predictions]  # 取第一个beam结果
        
        # 保存测试结果到文件
        output_dict = {}
        for idx, (pred, ex) in enumerate(zip(predictions, test_examples)):
            output_dict[str(ex.idx)] = (pred, ex.target)
        
        with open(os.path.join(args.output_dir, 'test_pred_gold.json'), 'w') as f:
            json.dump(output_dict, f, indent=2)
        
        # 使用bleu模块的computeMaps_multiple和bleuFromMaps方法计算BLEU分数
        (goldMap, predictionMap) = bleu.computeMaps_multiple(
            os.path.join(args.output_dir, 'test_pred_gold.json'),
            args.beam_size
        )
        bleu_score = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        
        # 记录测试BLEU分数到TensorBoard
        tb_writer.add_scalar('test/bleu', bleu_score, 0)
        
        # 记录一些测试示例到TensorBoard
        num_examples = min(10, len(predictions))
        for i in range(num_examples):
            tb_writer.add_text(
                f'test/example_{i}',
                f"**Source**: {tokenizer.decode(test_features[i].source_ids, skip_special_tokens=True)}\n\n"
                f"**Prediction (Top-1)**: {predictions[i][0]}\n\n"
                f"**Target**: {test_examples[i].target}",
                0
            )
        
        # 关闭TensorBoard写入器
        if not args.do_train:
            tb_writer.close()


if __name__ == '__main__':
    main() 