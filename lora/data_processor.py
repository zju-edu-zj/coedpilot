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
            target = js['target_tokens']  # 目标代码
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
                current_part = parts[i].strip()
                
                # 检查是否是remove操作
                if current_part.startswith("remove "):
                    removed_code = current_part[len("remove "):]
                    
                    # 检查下一个是否是add操作（形成一个完整的替换）
                    if i + 1 < len(parts) and parts[i+1].strip().startswith("add "):
                        added_code = parts[i+1].strip()[len("add "):]
                        prior_edits.append(f"REPLACED: \n{removed_code}\nWITH: \n{added_code}")
                        i += 2  # 跳过下一个add操作
                    else:
                        # 单独的remove操作
                        prior_edits.append(f"REMOVED: \n{removed_code}")
                        i += 1
                # 检查是否是add操作
                elif current_part.startswith("add "):
                    added_code = current_part[len("add "):]
                    prior_edits.append(f"ADDED: \n{added_code}")
                    i += 1
                else:
                    # 其他未知操作，直接添加
                    prior_edits.append(current_part)
                    i += 1
        
        prior_edits_text = "\n\n".join(prior_edits)
        
        # 构建提示模板
        if stage == 'test':
            prompt = (f"Given the following code context with edit markers and previous edits, "
                     f"generate the appropriate code:\n\n"
                     f"CODE CONTEXT:\n{processed_context}\n\n"
                     f"PREVIOUS EDITS:\n{prior_edits_text}\n\n"
                     f"GENERATED CODE:")
            target = ""
        else:
            prompt = (f"Given the following code context with edit markers and previous edits, "
                     f"generate the appropriate code:\n\n"
                     f"CODE CONTEXT:\n{processed_context}\n\n"
                     f"PREVIOUS EDITS:\n{prior_edits_text}\n\n"
                     f"GENERATED CODE:")
            target = example.target
        
        if stage == "pure":
            print(f"prompt: {prompt}")
            print(f"target: {target}")
            return

        tokenized_prompt = tokenizer(prompt, truncation=True, max_length=args.max_source_length)
        tokenized_target = tokenizer(target, truncation=True, max_length=args.max_target_length)
        
        source_ids = tokenized_prompt["input_ids"]
        target_ids = tokenized_target["input_ids"]
        
        source_mask = tokenized_prompt["attention_mask"]
        target_mask = tokenized_target["attention_mask"]
        
        if example_index < 1:
            if stage == 'train':
                logger.info('*** Example ***')
                logger.info('idx: {}'.format(example.idx))
                logger.info('prompt: {}'.format(prompt))
                logger.info('target: {}'.format(target))
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
    return features


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


def load_features(input_file):
    """从文件加载处理好的特征"""
    features = torch.load(input_file)
    logger.info(f"Loaded {len(features)} features from {input_file}")
    return features


def process_and_save_data(train_file, dev_file, test_file, tokenizer, args, output_dir):
    """处理所有数据集并保存"""
    import os
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理训练集
    if train_file:
        train_examples = read_examples(train_file)
        train_features = prepare_training_features(train_examples, tokenizer, args)
        save_features(train_features, os.path.join(output_dir, 'train_features.pt'))
    
    # 处理验证集
    if dev_file:
        dev_examples = read_examples(dev_file)
        dev_features = prepare_validation_features(dev_examples, tokenizer, args)
        save_features(dev_features, os.path.join(output_dir, 'dev_features.pt'))
    
    # 处理测试集
    if test_file:
        test_examples = read_examples(test_file)
        test_features = prepare_test_features(test_examples, tokenizer, args)
        save_features(test_features, os.path.join(output_dir, 'test_features.pt')) 

if __name__ == "__main__":
    examples = read_examples("new_test.jsonl")
    convert_examples_to_features(examples, None, None, stage="pure")
