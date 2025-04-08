import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
from tqdm import tqdm
import bleu  # 添加bleu模块导入

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

def batch_generate_responses(model, tokenizer, prompts, max_new_tokens=512, temperature=0.2, batch_size=4, beam_size=1):
    """批量生成响应"""
    all_responses = []
    
    # 确保pad_token_id已设置
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # 分批处理
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                do_sample=(temperature > 0),
                temperature=temperature,
                pad_token_id=model.config.pad_token_id,
                early_stopping=True,
                repetition_penalty=1.0,
                length_penalty=1.0,
                no_repeat_ngram_size=0,
            )
        
        # 重塑输出以获取每个输入的beam_size个结果
        generated_ids = generated_ids.reshape(len(batch_prompts), beam_size, -1)
        
        for i, beam_outputs in enumerate(generated_ids):
            beam_results = []
            input_length = len(inputs.input_ids[i])
            for beam_output in beam_outputs:
                gen_text = tokenizer.decode(beam_output[input_length:], skip_special_tokens=True)
                # 如果响应包含EOT标记，则截断
                if "<|EOT|>" in gen_text:
                    gen_text = gen_text.split("<|EOT|>")[0].strip()
                beam_results.append(gen_text)
            all_responses.append(beam_results)
    
    return all_responses

def main():
    parser = argparse.ArgumentParser(description="测试微调后的模型")
    parser.add_argument("--base_model", type=str, required=True, help="基础模型路径")
    parser.add_argument("--lora_model", type=str, required=True, help="LoRA模型路径")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据路径")
    parser.add_argument("--output_file", type=str, default="test_results.json", help="输出结果文件")
    parser.add_argument("--bleu_output", type=str, default="test_pred_gold.json", help="BLEU评估输出文件")
    parser.add_argument("--max_samples", type=int, default=None, help="最大测试样本数")
    parser.add_argument("--temperature", type=float, default=0.2, help="生成温度")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成token数")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小")
    parser.add_argument("--beam_size", type=int, default=5, help="束搜索大小")
    args = parser.parse_args()
    
    # 加载模型和分词器
    print(f"正在加载基础模型: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=True  # 启用KV缓存以加速生成
    )
    
    # 加载LoRA权重（如果有的话）
    if args.lora_model:
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
    prompts = []
    expected_codes = []
    print(f"开始测试 {len(test_data)} 个样本...")
    
    for example in test_data:
        # 构建提示
        instruction = process_source(example)
        prompt = build_instruction_prompt(instruction)
        prompts.append(prompt)
        expected_codes.append(example.get("docstring_tokens", ""))
    
    # 批量生成响应
    generated_responses = batch_generate_responses(
        model, 
        tokenizer, 
        prompts, 
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        beam_size=args.beam_size
    )
    
    # 保存结果
    results = []
    bleu_output_dict = {}
    
    for i, (prompt, responses, expected) in enumerate(zip(prompts, generated_responses, expected_codes)):
        result = {
            "id": i,
            "prompt": prompt,
            "generated_code": responses[0],  # 取第一个beam结果作为主要结果
            "all_generated": responses,      # 保存所有beam结果
            "expected_code": expected,
        }
        results.append(result)
        
        # 为BLEU评估准备数据
        bleu_output_dict[str(i)] = (responses, expected)
    
    # 保存详细结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # # 保存BLEU评估数据
    # bleu_output_path = os.path.join(os.path.dirname(args.output_file), args.bleu_output)
    # with open(bleu_output_path, 'w', encoding='utf-8') as f:
    #     json.dump(bleu_output_dict, f, ensure_ascii=False, indent=2)
    
    # # 计算BLEU分数
    # try:
    #     (goldMap, predictionMap) = bleu.computeMaps_multiple(bleu_output_path, args.beam_size)
    #     bleu_score = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    #     print(f"BLEU分数: {bleu_score}")
        
    #     # 将BLEU分数添加到结果文件中
    #     with open(os.path.join(os.path.dirname(args.output_file), "bleu_score.txt"), 'w') as f:
    #         f.write(f"BLEU分数: {bleu_score}\n")
    # except Exception as e:
    #     print(f"计算BLEU分数时出错: {e}")
    
    print(f"测试完成，结果已保存到 {args.output_file}")

if __name__ == "__main__":
    main() 