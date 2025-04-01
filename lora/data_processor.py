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

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

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
    
    # 确保tokenizer使用右侧填充，与DeepSeek模型一致
    logger.info(f"tokenizer padding side: {tokenizer.padding_side}")
    tokenizer.padding_side = 'right'
    
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
        
        source_text = processed_source
        
        # 目标文本添加EOT_TOKEN
        target_text = f"{example.target}\n{EOT_TOKEN}"
        
        if stage == 'train':
            # 使用与finetune_deepseek.py中相同的预处理逻辑
            # 1. 将source和target组合
            examples_text = source_text + target_text
            
            # 2. 分别tokenize组合文本和源文本
            source_tokens = tokenizer.encode(
                source_text,
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_source_length,
            )
            
            combined_tokens = tokenizer.encode(
                examples_text,
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_source_length + args.max_target_length,
            )
            
            # 3. 创建标签，将源文本部分设置为IGNORE_INDEX
            source_len = len(source_tokens)
            labels = [-100] * source_len + combined_tokens[source_len:]
            
            # 确保长度一致
            if len(labels) < len(combined_tokens):
                logger.info(f"labels长度小于combined_tokens，进行填充")
                labels = labels + [-100] * (len(combined_tokens) - len(labels))
            elif len(labels) > len(combined_tokens):
                logger.info(f"labels长度大于combined_tokens，进行截断")
                labels = labels[:len(combined_tokens)]
            
            # 4. 创建attention mask
            attention_mask = [1] * len(combined_tokens)
            
            # 5. 进行填充处理
            padding_length = args.max_source_length + args.max_target_length - len(combined_tokens)
            if padding_length > 0:
                combined_tokens = combined_tokens + [tokenizer.pad_token_id] * padding_length
                labels = labels + [-100] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            
            # 使用处理后的序列
            source_ids = combined_tokens
            source_mask = attention_mask
            target_ids = labels
            target_mask = attention_mask
        else:
            # 验证和测试阶段只需要处理输入
            source_inputs = tokenizer(
                source_text,
                truncation=True,
                max_length=args.max_source_length,
                padding='max_length',
                return_tensors=None
            )
            
            source_ids = source_inputs["input_ids"]
            source_mask = source_inputs["attention_mask"]
            
            # 为了保持一致性，仍然创建target_ids和target_mask
            target_inputs = tokenizer(
                target_text,
                truncation=True,
                max_length=args.max_target_length,
                padding='max_length',
                return_tensors=None
            )
            
            target_ids = target_inputs["input_ids"]
            target_mask = target_inputs["attention_mask"]
        
        # 记录示例信息
        if example_index < 1:
            logger.info('*** Example ***')
            logger.info('idx: {}'.format(example.idx))
            logger.info('source_text: {}'.format(source_text))
            logger.info('target_text: {}'.format(target_text))
            logger.info('source_ids: {}'.format(' '.join(map(str, source_ids))))
            logger.info('source_mask: {}'.format(' '.join(map(str, source_mask))))
            logger.info('target_ids: {}'.format(' '.join(map(str, target_ids))))
            logger.info('target_mask: {}'.format(' '.join(map(str, target_mask))))
        
        # 添加到features列表
        features.append(
            InputFeatures(
                example_index=example_index,
                source_ids=source_ids,
                target_ids=target_ids,
                source_mask=source_mask,
                target_mask=target_mask,
            )
        )
        
        # 记录提示信息
        prompts.append({
            "example_index": example_index,
            "source": source_text,
            "target": target_text
        })
    
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

# 添加build_instruction_prompt函数，与finetune_deepseek.py中保持一致
def build_instruction_prompt(instruction: str):
    return '''
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()

if __name__ == "__main__":
    examples = read_examples("new_test.jsonl")
    convert_examples_to_features(examples, None, None, stage="pure")
