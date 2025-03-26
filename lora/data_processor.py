# coding=utf-8
import json
import logging
import torch
from tqdm import tqdm

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


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(
        self, example_index, source_ids, target_ids, source_mask, target_mask
    ):
        self.example_index = example_index
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx

            code = js['code_tokens']  # 包含<mask>标记的代码
            target = js['docstring_tokens']  # 目标代码
            label_window = js['label_window']  # 替换<mask>的操作

            # 检查mask数量与label_window长度是否匹配
            num_mask = code.count('<mask>')
            if num_mask != len(label_window):
                continue
            examples.append(Example(idx=idx, source=code, target=target, edit_ops=label_window))
    return examples


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    """Convert examples to features that can be used for model training."""
    features = []
    prompts = []  # 用于存储所有提示
    
    for example_index, example in enumerate(
        tqdm(examples, desc='convert examples to features')
    ):
        # 解析输入数据
        source_code = example.source
        
        # 分离上下文和先前编辑
        parts = source_code.split("</s>")
        context = parts[0]  # 第一个</s>之前的部分是上下文
        
        # 处理上下文中的<mask>标记
        context_lines = context.split('\n')
        label_idx = 0
        processed_lines = []
        
        for line in context_lines:
            if "<mask>" in line:
                # 根据label_window中的操作类型替换<mask>
                if label_idx < len(example.edit_ops):
                    op = example.edit_ops[label_idx]
                    if op == "add":
                        # 标记为添加操作
                        processed_line = line.replace("<mask>", "<ADD_CODE>")
                    elif op == "replace":
                        # 标记为替换操作
                        processed_line = line.replace("<mask>", "<REPLACE_CODE>")
                    else:  # keep或其他操作
                        # 将keep也设置为特殊标记
                        processed_line = line.replace("<mask>", "<KEEP_CODE>")
                    label_idx += 1
                else:
                    # 如果没有对应的操作，默认保持不变
                    processed_line = line.replace("<mask>", "<KEEP_CODE>")
                processed_lines.append(processed_line)
            else:
                processed_lines.append(line)
        
        processed_context = '\n'.join(processed_lines)
        
        # 处理先前的编辑，将开头的add和remove替换为特殊标记
        processed_parts = [processed_context]  # 第一部分是处理后的上下文
        
        if len(parts) > 1:
            for i in range(1, len(parts)):
                part = parts[i]
                # 检查是否以"add "开头
                if part.lstrip().startswith("add "):
                    processed_part = part.replace("add ", "<ADD> ", 1)  # 只替换第一次出现
                # 检查是否以"remove "开头
                elif part.lstrip().startswith("remove "):
                    processed_part = part.replace("remove ", "<REMOVE> ", 1)  # 只替换第一次出现
                else:
                    processed_part = part
                processed_parts.append(processed_part)
        
        # 将处理后的部分重新组合，使用tokenizer的eos_token替换</s>
        processed_source = tokenizer.eos_token.join(processed_parts)
        
        # 使用tokenizer直接处理源代码和目标
        source_inputs = tokenizer(
            processed_source,
            truncation=True,
            max_length=args.max_source_length,
            padding='max_length',
            return_tensors=None  # 返回列表而不是张量
        )
        
        source_ids = source_inputs["input_ids"]
        source_mask = source_inputs["attention_mask"]
        
        # 处理目标
        if stage == 'test':
            target_text = 'None'
        else:
            target_text = example.target
            
        target_inputs = tokenizer(
            target_text,
            truncation=True,
            max_length=args.max_target_length,
            padding='max_length',
            return_tensors=None  # 返回列表而不是张量
        )
        
        target_ids = target_inputs["input_ids"]
        target_mask = target_inputs["attention_mask"]
        
        # 记录提示信息
        prompts.append({
            "example_index": example_index,
            "source": processed_source,
            "target": example.target
        })
        
        if example_index < 1:
            if stage == 'train':
                logger.info('*** Example ***')
                logger.info('idx: {}'.format(example.idx))
                logger.info('source_ids: {}'.format(' '.join(map(str, source_ids))))
                logger.info('source_mask: {}'.format(' '.join(map(str, source_mask))))
                logger.info('target_ids: {}'.format(' '.join(map(str, target_ids))))
                logger.info('target_mask: {}'.format(' '.join(map(str, target_mask))))
        
        features.append(
            InputFeatures(
                example_index=example_index,
                source_ids=source_ids,
                target_ids=target_ids,
                source_mask=source_mask,
                target_mask=target_mask,
            )
        )
    
    return features, prompts


def prepare_training_features(examples, tokenizer, args):
    """Prepare features for training."""
    return convert_examples_to_features(examples, tokenizer, args, stage='train')


def prepare_validation_features(examples, tokenizer, args):
    """Prepare features for validation."""
    return convert_examples_to_features(examples, tokenizer, args, stage='dev')


def prepare_test_features(examples, tokenizer, args):
    """Prepare features for testing."""
    return convert_examples_to_features(examples, tokenizer, args, stage='test')


def save_features(features, output_file):
    """将处理好的特征保存到文件"""
    torch.save(features, output_file)
    logger.info(f"Features saved to {output_file}")


def save_prompts(prompts, output_file):
    """将提示信息保存到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    logger.info(f"Prompts saved to {output_file}")


def load_features(input_file):
    """从文件加载处理好的特征"""
    features = torch.load(input_file)
    logger.info(f"Loaded {len(features)} features from {input_file}")
    return features


def load_prompts(input_file):
    """从文件加载处理好的提示信息"""
    with open(input_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    logger.info(f"Loaded {len(prompts)} prompts from {input_file}")
    return prompts


def process_and_save_data(train_file, dev_file, test_file, tokenizer, args, output_dir):
    """处理所有数据集并保存"""
    import os
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理训练集
    if train_file:
        train_examples = read_examples(train_file)
        train_features, train_prompts = prepare_training_features(train_examples, tokenizer, args)
        save_features(train_features, os.path.join(output_dir, 'train_features.pt'))
        save_prompts(train_prompts, os.path.join(output_dir, 'train_prompts.json'))
    
    # 处理验证集
    if dev_file:
        dev_examples = read_examples(dev_file)
        dev_features, dev_prompts = prepare_validation_features(dev_examples, tokenizer, args)
        save_features(dev_features, os.path.join(output_dir, 'dev_features.pt'))
        save_prompts(dev_prompts, os.path.join(output_dir, 'dev_prompts.json'))
    
    # 处理测试集
    if test_file:
        test_examples = read_examples(test_file)
        test_features, test_prompts = prepare_test_features(test_examples, tokenizer, args)
        save_features(test_features, os.path.join(output_dir, 'test_features.pt'))
        save_prompts(test_prompts, os.path.join(output_dir, 'test_prompts.json'))

if __name__ == "__main__":
    examples = read_examples("new_test.jsonl")
    convert_examples_to_features(examples, None, None, stage="pure")
