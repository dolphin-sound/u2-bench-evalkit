#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量测试脚本 - 阿里云百炼平台版本
使用role_play_model_test_reinforced.py中的函数测试多个问题
"""
import json
import os
import time
import argparse
import random
import multiprocessing
import asyncio
import aiohttp
import base64
import requests
import sys
from datetime import datetime
from openai import OpenAI
import pandas as pd

# 添加父目录到系统路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.dirname(parent_dir))

# API配置常量
DEFAULT_API_KEY = "YOUR API KEY"
API_RETRIES = 3
EMPTY_RESPONSE_RETRIES = 3
MIN_VALID_RESPONSE_LENGTH = 1

def encode_image(image_path):
    """
    将图片转换为Base64编码
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        Base64编码的图片数据
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"图片编码失败: {str(e)}")
        return None

def call_model_multimodal(model_name, messages, max_tokens=800, temperature=0.7, retries=API_RETRIES, 
                         api_key=DEFAULT_API_KEY, empty_retries=EMPTY_RESPONSE_RETRIES):
    """
    调用百炼平台模型API获取多模态输入（图像+文本）的响应
    
    Args:
        model_name: 模型名称
        messages: 多模态消息列表，每个消息是包含type和value字段的字典
                 例如: [{"type": "image", "value": "图像路径"}, {"type": "text", "value": "文本内容"}]
        max_tokens: 最大生成token数
        temperature: 温度参数
        retries: API调用失败时的最大重试次数
        api_key: API密钥
        empty_retries: 空回复的最大重试次数
        
    Returns:
        模型响应文本
    """
    # 初始化百炼客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # 构建OpenAI格式的消息
    openai_messages = []
    
    # 处理多模态消息
    for msg in messages:
        if msg["type"] == "image":
            # 处理图像
            image_path = msg["value"]
            if os.path.exists(image_path):
                # 编码图像
                image_base64 = encode_image(image_path)
                if image_base64:
                    # 添加图像消息
                    openai_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    })
            else:
                print(f"警告: 图像文件不存在: {image_path}")
        
        elif msg["type"] == "text":
            # 处理文本
            text_content = msg["value"]
            openai_messages.append({
                "role": "user",
                "content": text_content
            })
    
    # 如果没有有效消息，返回错误
    if not openai_messages:
        return "错误: 没有有效的消息内容"
    
    # 尝试调用API
    for attempt in range(retries):
        try:
            # 调用API
            response = client.chat.completions.create(
                model=model_name,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # 处理响应
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                response_text = response.choices[0].message.content
                
                # 检查响应是否为空或太短
                if not response_text or len(response_text.strip()) < MIN_VALID_RESPONSE_LENGTH:
                    if attempt < empty_retries - 1:
                        print(f"警告: 收到空响应，重试 ({attempt+1}/{empty_retries})")
                        continue
                    else:
                        return "API返回了空响应"
                
                return response_text
            else:
                if attempt < retries - 1:
                    print(f"警告: API响应格式错误，重试 ({attempt+1}/{retries})")
                    continue
                else:
                    return "API响应格式错误"
                
        except Exception as e:
            if attempt < retries - 1:
                print(f"警告: API调用失败，重试 ({attempt+1}/{retries}): {str(e)}")
                time.sleep(1)  # 短暂延迟后重试
                continue
            else:
                return f"API调用失败: {str(e)}"
    
    return "多次API调用尝试失败"

def run_batch_test(task_type, dataset_name, output_dir, model_name, api_key=DEFAULT_API_KEY, sample_limit=None):
    """
    运行单个任务的批量测试
    
    Args:
        task_type: 任务类型 ('cla', 'measurement', 'seg', 'report')
        dataset_name: 数据集名称
        output_dir: 输出目录
        model_name: 模型名称
        api_key: API密钥
        sample_limit: 限制每个数据集的样本数量
    """
    # 导入数据集
    if task_type == 'cla':
        from vlmeval.dataset.image_cla import ClaDataset
        dataset = ClaDataset(dataset=dataset_name)
    elif task_type == 'measurement':
        from vlmeval.dataset.image_measurement import MeasureDataset
        dataset = MeasureDataset(dataset=dataset_name)
    elif task_type == 'seg':
        from vlmeval.dataset.image_seg import SegDataset
        dataset = SegDataset(dataset=dataset_name)
    elif task_type == 'report':
        from vlmeval.dataset.image_report import ReportDataset
        dataset = ReportDataset(dataset=dataset_name)
    else:
        print(f"不支持的任务类型: {task_type}")
        return

    # 获取样本数量
    total_samples = len(dataset)
    if sample_limit and sample_limit > 0:
        total_samples = min(sample_limit, total_samples)

    # 创建输出目录
    task_output_dir = os.path.join(output_dir, task_type, dataset_name)
    os.makedirs(task_output_dir, exist_ok=True)

    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(task_output_dir, f"async_multimodal_test_results_{timestamp}.jsonl")
    
    # 创建存储所有结果的列表
    all_results = []

    # 处理样本
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(total_samples):
            # 构建提示
            prompt = dataset.build_prompt(i, task_type)
            
            # 调用模型
            response = call_model_multimodal(
                model_name=model_name,
                messages=prompt,
                max_tokens=800,
                temperature=0.7,
                api_key=api_key
            )
            
            # 构建结果记录
            result = {
                'id': str(i),
                'text_content': next((msg['value'] for msg in prompt if msg['type'] == 'text'), ''),
                'image_paths': [msg['value'] for msg in prompt if msg['type'] == 'image'],
                'category': task_type,
                'response': response,
                'model': model_name,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 添加到结果列表
            all_results.append(result)
            
            # 写入jsonl文件
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()  # 确保实时写入
            
            print(f"已处理样本 {i+1}/{total_samples}")
        
        print(f"已将所有结果保存到: {output_file}")
    
    # 创建合并结果文件
    combined_output_file = os.path.join(task_output_dir, f"{model_name}_combined_results.jsonl")
    with open(combined_output_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"已将所有结果合并保存到: {combined_output_file}")

def run_multi_task_test(task_types, output_dir, model_name, api_key=DEFAULT_API_KEY, test_all_datasets=False, sample_limit=None):
    """
    运行多任务测试
    
    Args:
        task_types: 任务类型列表
        output_dir: 输出目录
        model_name: 模型名称
        api_key: API密钥
        test_all_datasets: 是否测试所有数据集
        sample_limit: 限制每个数据集的样本数量
    """
    # 定义任务和数据集
    tasks_datasets = {
        'cla': ['03', '10', '18', '21', '23', '25', '28', '32', '37', '40', '42', '44', '50', '53', '57', '66', '69', '70', '74is_normal', '74is_visible', '75', 'anatomy'],
        'seg': ['04', '09', '13', '16', '17', '18', '23', '31', '32', '37', '38', '47', '48', '49', '50', '52', '53', '64', '67'],
        'measurement': ['18', '27', '31', '50', '57'],
        'report': ['10', '39', '44']
    }
    
    for task_type in task_types:
        if task_type in tasks_datasets:
            if test_all_datasets:
                # 测试该任务类型下的所有数据集
                for dataset_name in tasks_datasets[task_type]:
                    run_batch_test(task_type, dataset_name, output_dir, model_name, api_key, sample_limit)
            else:
                # 只测试第一个数据集
                dataset_name = tasks_datasets[task_type][0]
                run_batch_test(task_type, dataset_name, output_dir, model_name, api_key, sample_limit)

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='批量测试脚本 - 阿里云百炼平台版本')
    parser.add_argument('--task_types', type=str, nargs='+',
                        choices=['cla', 'measurement', 'seg', 'report'],
                        help='任务类型列表')
    parser.add_argument('--task_type', type=str,
                        choices=['cla', 'measurement', 'seg', 'report'],
                        help='单个任务类型')
    parser.add_argument('--dataset_name', type=str,
                        help='数据集名称，与task_type一起使用')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='输出目录')
    parser.add_argument('--test_all_datasets', action='store_true',
                        help='是否测试所有数据集')
    parser.add_argument('--sample_limit', type=int, default=None,
                        help='限制每个数据集的样本数量')
    parser.add_argument('--model', type=str, default="qwen2.5-vl-3b-instruct",
                        help='模型名称')
    parser.add_argument('--api_key', type=str, default=DEFAULT_API_KEY,
                        help='API密钥')
    parser.add_argument('--multimodal', action='store_true',
                        help='是否使用多模态模式（直接从数据集加载多模态问题）')
    
    args = parser.parse_args()
    
    # 如果指定了单个任务类型和数据集，直接运行单个测试
    if args.task_type and args.dataset_name:
        print(f"运行单个测试: {args.task_type}/{args.dataset_name}，模型: {args.model}")
        run_batch_test(
            task_type=args.task_type,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            model_name=args.model,
            api_key=args.api_key,
            sample_limit=args.sample_limit
        )
    # 否则运行多任务测试
    elif args.task_types:
        run_multi_task_test(
            task_types=args.task_types,
            output_dir=args.output_dir,
            model_name=args.model,
            api_key=args.api_key,
            test_all_datasets=args.test_all_datasets,
            sample_limit=args.sample_limit
        )
    else:
        print("错误: 必须指定task_type和dataset_name，或者指定task_types")
        parser.print_help()

if __name__ == "__main__":
    main()
