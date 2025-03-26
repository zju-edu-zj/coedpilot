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
    prompts = []  # 新增：用于存储所有提示
    
    # 收集所有提示和目标，以便后续批量处理
    all_prompts = []
    all_targets = []
    all_example_indices = []
    
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
                        # 保持不变，移除<mask>
                        processed_line = line.replace("<mask>", "")
                    label_idx += 1
                else:
                    # 如果没有对应的操作，默认保持不变
                    processed_line = line.replace("<mask>", "")
                processed_lines.append(processed_line)
            else:
                processed_lines.append(line)
        
        processed_context = '\n'.join(processed_lines)
        
        # 收集先前的编辑信息，将连续的remove和add视为一个编辑
        prior_edits = []
        if len(parts) > 1:
            i = 1
            while i < len(parts):
                current_part = parts[i].lstrip()  # 先只去除左侧空白
                
                # 检查是否是remove操作
                if current_part.startswith("remove "):
                    removed_code = current_part[len("remove "):].rstrip()  # 提取后再去除右侧空白
                    
                    # 检查下一个是否是add操作（形成一个完整的替换）
                    if i + 1 < len(parts) and parts[i+1].lstrip().startswith("add "):
                        added_code = parts[i+1].lstrip()[len("add "):].rstrip()
                        if added_code.strip():
                            prior_edits.append(f"REPLACED:\n```python\n{removed_code}\n```\nWITH:\n```python\n{added_code}\n```")
                        else:
                            prior_edits.append(f"REPLACED:\n```python\n{removed_code}\n```\nWITH: [EMPTY]")
                        i += 2  # 跳过下一个add操作
                    else:
                        # 单独的remove操作
                        prior_edits.append(f"REMOVED:\n```python\n{removed_code}\n```")
                        i += 1
                # 检查是否是add操作
                elif current_part.startswith("add "):
                    added_code = current_part[len("add "):].rstrip()  # 提取后再去除右侧空白
                    prior_edits.append(f"ADDED:\n```python\n{added_code}\n```")
                    i += 1
                else:
                    # 保留PR描述和其他信息
                    prior_edits.append(current_part.rstrip())
                    i += 1
        
        prior_edits_text = "\n\n".join(prior_edits)
        
        # 构建提示模板
        prompt = (f"Code context:\n{processed_context}\n\n"
                 f"Previous edits:\n{prior_edits_text}\n\n"
                 f"Output:")
        
        if stage == 'test':
            target = ""
        else:
            target = example.target
        
        # 收集提示和目标
        all_prompts.append(prompt)
        all_targets.append(target)
        all_example_indices.append(example_index)
        
        # 保存提示信息
        prompts.append({
            "example_index": example_index,
            "prompt": prompt,
            "target": target
        })
        
        if stage == "pure":
            print(f"prompt: {prompt}")
            print(f"target: {target}")
            return

    if stage == "pure":
        print(f"prompt: {all_prompts[0]}")
        print(f"target: {all_targets[0]}")
        return
    
    # 批量处理所有提示和目标，启用填充
    tokenized_prompts = tokenizer(
        all_prompts, 
        truncation=True, 
        max_length=args.max_source_length,
        padding='max_length'  # 确保所有序列长度一致
    )
    
    tokenized_targets = tokenizer(
        all_targets, 
        truncation=True, 
        max_length=args.max_target_length,
        padding='max_length'  # 确保所有序列长度一致
    )
    
    # 创建特征
    for i, example_index in enumerate(all_example_indices):
        source_ids = tokenized_prompts["input_ids"][i]
        target_ids = tokenized_targets["input_ids"][i]
        
        source_mask = tokenized_prompts["attention_mask"][i]
        target_mask = tokenized_targets["attention_mask"][i]
        
        if i < 1:
            if stage == 'train':
                logger.info('*** Example ***')
                logger.info('idx: {}'.format(examples[example_index].idx))
                logger.info('prompt: {}'.format(all_prompts[i]))
                logger.info('target: {}'.format(all_targets[i]))
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
    
    # 验证所有特征长度一致
    if features:
        source_lengths = [len(f.source_ids) for f in features]
        target_lengths = [len(f.target_ids) for f in features]
        logger.info(f"Stage: {stage}, Source lengths: min={min(source_lengths)}, max={max(source_lengths)}")
        logger.info(f"Stage: {stage}, Target lengths: min={min(target_lengths)}, max={max(target_lengths)}")
    
    # 返回提示信息和特征
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
