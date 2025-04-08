import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed
import transformers
from transformers import Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import json
import pandas as pd
from datasets import Dataset


IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"


def process_source(example):
    """处理源代码，先替换<mask>标签，然后处理编辑操作"""
    # 首先用edit_ops替换源代码中的<mask>
    source_code = example['code_tokens']
    edit_ops = example['label_window']
    
    # 确保edit_ops是列表形式
    if not isinstance(edit_ops, list):
        edit_ops = [edit_ops]
    
    # 替换每个<mask>为对应的编辑操作
    for edit_op in edit_ops:
        source_code = source_code.replace('<mask>', edit_op, 1)  # 只替换第一个匹配项
    
    # 分割处理后的代码
    parts = source_code.split('</s>')
    code_window = parts[0].strip()
    
    # 解析代码窗口中的每一行
    code_lines = code_window.split('\n')
    add_positions = []
    replace_positions = []
    context_lines = []
    
    for i, line in enumerate(code_lines):
        line = line.lstrip()
        if not line:  # 跳过空行
            continue
        if line.startswith('add '):
            # 记录需要插入代码的位置和上下文
            add_positions.append(i)
            remaining = line[4:]
            context_lines.append(remaining)
            
        elif line.startswith('replace '):
            # 记录需要替换的位置和上下文
            replace_positions.append(i)
            remaining = line[8:]
            context_lines.append(remaining)
        elif line.startswith('keep '):
            # 保留上下文信息
            remaining = line[5:]
            context_lines.append(remaining)
    
    # 提取commit message和历史编辑
    commit_message = parts[1].strip() if len(parts) > 1 else ""
    # 如果commit message太长，只取第一行
    if commit_message and len(commit_message) > 100:
        commit_message = commit_message.split("\n")[0]
    
    # 历史编辑操作作为一个集合处理
    edit_history = set()
    if len(parts) > 2:
        for edit in parts[2:]:
            edit_history.add(edit)
    
    # 构建prompt
    operations = []
    if add_positions:
        operations.append(f"Add code at line(s): {', '.join(map(str, add_positions))}")
    if replace_positions:
        operations.append(f"Replace code at line(s): {', '.join(map(str, replace_positions))}")
    
    prompt = f"""Generate code based on the following information:

Code Context:
{'-' * 40}
{chr(10).join(context_lines)}
{'-' * 40}

Required Operations:
{chr(10).join(operations)}

Description:
{commit_message}

Related Historical Edits:
{chr(10).join(f"• {edit}" for edit in edit_history) if edit_history else 'None'}

"""
    
    return prompt

def build_instruction_prompt(instruction: str):
    return '''
You are a professional code editing assistant. Please generate only the code snippet to be inserted based on the provided information without any explanations or additional text.
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="deepseek-ai/deepseek-coder-6.7b-instruct")
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA for fine-tuning"})
    lora_r: int = field(default=8, metadata={"help": "LoRA attention dimension"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout value"})
    target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules to apply LoRA to"}
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    do_train: bool = field(default=True, metadata={"help": "Whether to perform training."})

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer):
    sources = []
    for idx in range(len(examples['code_tokens'])):
        # 构建每个样本的指令
        example = {
            'code_tokens': examples['code_tokens'][idx],
            'label_window': examples['label_window'][idx]
        }
        instruction = process_source(example)
        sources.append(build_instruction_prompt(instruction))
    
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['docstring_tokens']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

class ProgressCallback(TrainerCallback):
    """自定义回调函数，用于打印训练进度"""
    def __init__(self):
        self.last_log = 0
        self.log_interval = 10  # 每10个步骤打印一次

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step - self.last_log >= self.log_interval:
            # 计算进度百分比
            progress = state.global_step / state.max_steps * 100
            loss = state.log_history[-1].get("loss", 0) if state.log_history else 0
            
            print(f"训练进度: {progress:.2f}% ({state.global_step}/{state.max_steps}) - Loss: {loss:.4f}")
            self.last_log = state.global_step

def load_jsonl_file(file_path):
    """手动加载JSONL文件并返回Dataset对象"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"警告：跳过无效的JSON行: {e}")
                continue
    
    # 转换为pandas DataFrame，然后转换为Dataset
    df = pd.DataFrame(data)
    return Dataset.from_pandas(df)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if training_args.local_rank == 0:
        print('='*100)
        print(training_args)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    if training_args.local_rank == 0:
        print("Load tokenizer from {} over.".format(model_args.model_name_or_path))

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16
    )

    if training_args.local_rank == 0:
        print("Load model from {} over.".format(model_args.model_name_or_path))
    
    # 添加LoRA配置
    if model_args.use_lora:
        if training_args.local_rank == 0:
            print("使用LoRA进行参数高效微调")
        
        # 准备模型进行LoRA微调
        target_modules = model_args.target_modules.split(",") if model_args.target_modules else None
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
        )
        
        # 将模型转换为LoRA模型
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # 打印可训练参数比例

    # 手动加载训练数据集
    print(f"正在加载训练数据: {data_args.data_path}")
    raw_train_datasets = load_jsonl_file(data_args.data_path)
    
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running Encoding",
        fn_kwargs={"tokenizer": tokenizer}
    )
    
    if training_args.local_rank == 0:
        print("训练数据集样本数:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), min(3, len(train_dataset))):
            print(f"训练集样本 {index}: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            print(f"训练集样本 {index} 解码: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, data_collator=data_collator)

    # 创建进度回调函数
    progress_callback = ProgressCallback()
    
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        callbacks=[progress_callback],  # 添加回调函数
        **data_module
    )

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()