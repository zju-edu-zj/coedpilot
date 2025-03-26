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

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    special_tokens = {
        "additional_special_tokens": ["<ADD_CODE>", "<REPLACE_CODE>"]
    }
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    logger.info(f"Added {num_added_tokens} new special tokens")
    
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
        device_map="auto"
    )
    
    # 为LoRA准备模型
    if load_in_8bit or load_in_4bit:
        logger.info("prepare model for kbit training", load_in_8bit, load_in_4bit)
        model = prepare_model_for_kbit_training(model)
    
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
    
    # 然后在加载模型后调整嵌入大小
    model.resize_token_embeddings(len(tokenizer))
    
    if args.do_train:
        if args.processed_data_dir:
            # 从处理好的文件加载特征
            train_features = load_features(os.path.join(args.processed_data_dir, 'train_features.pt'))
            logger.info(f"Loaded {len(train_features)} training features")
        else:
            # 从原始文件处理特征
            logger.info("unexpected, please save processed data first")
            exit()
        
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
                
                bar.set_postfix(loss=epoch_loss/(step+1))
            
            tr_loss += epoch_loss
            
            # 保存每个epoch的模型
            output_dir = os.path.join(args.output_dir, f'checkpoint-epoch-{epoch}')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # 评估模型
            if args.do_eval:
                if args.processed_data_dir:
                    eval_features = load_features(os.path.join(args.processed_data_dir, 'dev_features.pt'))
                    # 需要加载原始示例用于评估
                    eval_examples = read_examples(args.dev_filename)
                else:
                    logger.info("unexpected, please save processed data first")
                    exit()
                
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
                
                model.eval()
                predictions = []
                
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
                        )
                    
                    for i, g_ids in enumerate(generated_ids):
                        # 找到输入结束的位置
                        input_length = len(input_ids[i])
                        gen_text = tokenizer.decode(g_ids[input_length:], skip_special_tokens=True)
                        predictions.append(gen_text)
                
                # 计算BLEU分数
                references = [ex.target for ex in eval_examples]
                bleu_score = bleu.compute_bleu(references, predictions)
                logger.info(f"BLEU score: {bleu_score}")
                
                # 保存最佳模型
                if bleu_score > best_bleu:
                    best_bleu = bleu_score
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                
                model.train()
        
        # 保存最终模型
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    
    # 测试模型
    if args.do_test:
        if args.processed_data_dir:
            test_features = load_features(os.path.join(args.processed_data_dir, 'test_features.pt'))
            test_examples = read_examples(args.test_filename)
        else:
            # 直接处理测试数据
            logger.info("处理测试数据...")
            test_examples = read_examples(args.test_filename)
            test_features = convert_examples_to_features(
                examples=test_examples,
                tokenizer=tokenizer,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length,
                stage='test'
            )
            logger.info(f"处理了 {len(test_features)} 条测试特征")
        
        # 加载最佳模型
        best_model_path = os.path.join(args.output_dir, 'checkpoint-best-bleu')
        if os.path.exists(best_model_path):
            model = AutoModelForCausalLM.from_pretrained(
                best_model_path,
                trust_remote_code=True,
                torch_dtype=compute_dtype,
                device_map="auto"
            )
        else:
            logger.info("找不到最佳模型，使用预训练模型")
        all_input_ids = torch.tensor([f.source_ids for f in test_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.source_mask for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_attention_mask)
        
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=args.eval_batch_size
        )
        
        logger.info("***** Running testing *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        model.eval()
        predictions = []
        
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
                )
            
            for i, beam_outputs in enumerate(generated_ids.reshape(input_ids.shape[0], args.beam_size, -1)):
                beam_results = []
                for beam_output in beam_outputs:
                    # 找到输入结束的位置
                    input_length = len(input_ids[i])
                    gen_text = tokenizer.decode(beam_output[input_length:], skip_special_tokens=True)
                    beam_results.append(gen_text)
                predictions.append(beam_results)
        
        # 保存测试结果
        output_dict = {}
        for idx, (pred, ex) in enumerate(zip(predictions, test_examples)):
            output_dict[str(ex.idx)] = (pred, ex.target)
        
        with open(os.path.join(args.output_dir, 'test_pred_gold.json'), 'w') as f:
            json.dump(output_dict, f, indent=2)
        
        # 计算BLEU分数
        references = [ex.target for ex in test_examples]
        best_predictions = [p[0] for p in predictions]  # 取第一个beam结果
        bleu_score = bleu.compute_bleu(references, best_predictions)
        logger.info(f"Test BLEU score: {bleu_score}")


if __name__ == '__main__':
    main() 