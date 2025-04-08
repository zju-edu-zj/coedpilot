import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
from tqdm import tqdm

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
You are a professional code editing assistant. You are a professional code editing assistant. Please generate only the code snippet to be inserted based on the provided information without any explanations or additional text.
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()

def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.2):
    """使用模型生成响应"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取响应部分（去除提示部分）
    response = response[len(prompt):].strip()
    
    # 如果响应包含EOT标记，则截断
    if "<|EOT|>" in response:
        response = response.split("<|EOT|>")[0].strip()
    
    return response

def main():
    parser = argparse.ArgumentParser(description="测试微调后的模型")
    parser.add_argument("--base_model", type=str, required=True, help="基础模型路径")
    parser.add_argument("--lora_model", type=str, required=True, help="LoRA模型路径")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据路径")
    parser.add_argument("--output_file", type=str, default="test_results.json", help="输出结果文件")
    parser.add_argument("--max_samples", type=int, default=None, help="最大测试样本数")
    parser.add_argument("--temperature", type=float, default=0.2, help="生成温度")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成token数")
    args = parser.parse_args()
    
    # 加载模型和分词器
    print(f"正在加载基础模型: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载LoRA权重
    print(f"正在加载LoRA权重: {args.lora_model}")
    model = PeftModel.from_pretrained(model, args.lora_model)
    model.eval()
    
    # 加载测试数据 (JSONL格式)
    print(f"正在加载测试数据: {args.test_data}")
    test_data = []
    with open(args.test_data, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                test_data.append(json.loads(line))
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 进行测试
    results = []
    print(f"开始测试 {len(test_data)} 个样本...")
    
    for i, example in enumerate(tqdm(test_data)):
        # 构建提示
        instruction = process_source(example)
        prompt = build_instruction_prompt(instruction)
        
        # 生成响应
        generated_code = generate_response(
            model, 
            tokenizer, 
            prompt, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
        # 保存结果
        result = {
            "id": i,
            "prompt": prompt,
            "generated_code": generated_code,
            "expected_code": example.get("docstring_tokens", ""),
        }
        results.append(result)
    
    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"测试完成，结果已保存到 {args.output_file}")

if __name__ == "__main__":
    main() 