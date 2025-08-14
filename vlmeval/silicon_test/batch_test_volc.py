#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量测试脚本 - 火山引擎方舟平台版本
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
from volcenginesdkarkruntime import Ark
import pandas as pd

# 添加父目录到系统路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.dirname(parent_dir))

# 导入数据集类
try:
    from vlmeval.dataset.dolphin.silicon_test.datasets import ClaDataset, MeasureDataset, SegDataset, ReportDataset
except ImportError:
    # 如果无法直接导入，尝试使用相对路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from vlmeval.dataset.dolphin.silicon_test.datasets import ClaDataset, MeasureDataset, SegDataset, ReportDataset
    except ImportError:
        print("无法导入数据集类，将使用文件路径方式加载数据")
        ClaDataset = MeasureDataset = SegDataset = ReportDataset = None

# API默认配置
DEFAULT_MODEL_NAME = "doubao-1-5-vision-pro-32k-250115"
DEFAULT_API_KEY = "YOUR API KEY"
API_RETRIES = 3
EMPTY_RESPONSE_RETRIES = 3
MIN_VALID_RESPONSE_LENGTH = 0

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
                         model_config=None, empty_retries=EMPTY_RESPONSE_RETRIES, api_key=DEFAULT_API_KEY):
    """
    调用方舟平台模型API获取多模态输入（图像+文本）的响应
    
    Args:
        model_name: 模型名称
        messages: 多模态消息列表，每个消息是包含type和value字段的字典
                 例如: [{"type": "image", "value": "图像路径"}, {"type": "text", "value": "文本内容"}]
        max_tokens: 最大生成token数
        temperature: 温度参数
        retries: API调用失败时的最大重试次数
        model_config: 模型配置
        empty_retries: 空回复的最大重试次数
        api_key: API密钥
        
    Returns:
        模型响应文本
    """
    # 初始化方舟客户端
    client = Ark(api_key=api_key)
    
    # 处理消息 - 将自定义格式转换为火山引擎SDK需要的格式
    # 火山引擎需要的格式: [{"role": "user", "content": [{"type": "text", "text": "文本内容"}, {"type": "image_url", "image_url": {"url": "图像URL"}}]}]
    
    # 首先，将所有消息合并到一个用户消息的content列表中
    content_list = []
    for msg in messages:
        if msg["type"] == "image":
            # 处理图像消息
            image_path = msg["value"]
            if not os.path.exists(image_path):
                print(f"图像文件不存在: {image_path}")
                return f"API调用失败: 图像文件不存在: {image_path}"
            
            # 编码图像
            image_base64 = encode_image(image_path)
            if not image_base64:
                return f"API调用失败: 图像编码失败: {image_path}"
            
            content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            })
        elif msg["type"] == "text":
            # 处理文本消息
            content_list.append({
                "type": "text",
                "text": msg["value"]
            })
    
    # 创建一个用户消息，包含所有内容
    processed_messages = [
        {
            "role": "user",
            "content": content_list
        }
    ]
    
    # 尝试调用API
    for attempt in range(retries):
        try:
            print(f"调用API，尝试 {attempt + 1}/{retries}...")
            # 使用火山引擎SDK的正确调用方式
            response = client.chat.completions.create(
                model=model_name,
                messages=processed_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **model_config if model_config else {}
            )
            
            # 检查响应
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
                
                # 检查响应是否为空
                if not content or len(content.strip()) < MIN_VALID_RESPONSE_LENGTH:
                    if attempt < empty_retries - 1:
                        print(f"收到空响应，重试中...")
                        time.sleep(2)  # 等待一段时间再重试
                        continue
                    else:
                        return "API调用失败: 收到空响应"
                
                return content
            else:
                print(f"API响应格式错误: {response}")
                if attempt < retries - 1:
                    time.sleep(2)  # 等待一段时间再重试
                    continue
                else:
                    return f"API调用失败: 响应格式错误: {response}"
        
        except Exception as e:
            print(f"API调用异常: {str(e)}")
            if attempt < retries - 1:
                time.sleep(2)  # 等待一段时间再重试
                continue
            else:
                return f"API调用失败: {str(e)}"
    
    return "API调用失败: 超过最大重试次数"

def save_question_result(result, model_name, questions_file_path, output_dir, timestamp=None):
    """
    保存单个问题的结果
    
    Args:
        result: 结果字典
        model_name: 模型名称
        questions_file_path: 问题文件路径
        output_dir: 输出目录
        timestamp: 时间戳
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取问题文件名（不包含扩展名）
    questions_file_name = os.path.splitext(os.path.basename(questions_file_path))[0]
    
    # 构建输出文件路径
    output_file = os.path.join(output_dir, f"async_multimodal_test_results_{timestamp}.jsonl")
    
    # 添加模型名称和时间戳到结果
    result["model"] = model_name
    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 将结果写入文件
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # 创建合并结果文件
    combined_output_file = os.path.join(output_dir, f"{model_name}_combined_results.jsonl")
    with open(combined_output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"已保存结果到 {output_file}")

def load_task_dataset(task_type, dataset_name):
    """
    加载指定任务类型和数据集的数据
    
    Args:
        task_type: 任务类型，可选值: 'cla', 'measurement', 'seg', 'report'
        dataset_name: 数据集名称
        
    Returns:
        数据集对象
    """
    try:
        if task_type == 'cla' and ClaDataset is not None:
            return ClaDataset(dataset=dataset_name)
        elif task_type == 'measurement' and MeasureDataset is not None:
            return MeasureDataset(dataset=dataset_name)
        elif task_type == 'seg' and SegDataset is not None:
            return SegDataset(dataset=dataset_name)
        elif task_type == 'report' and ReportDataset is not None:
            return ReportDataset(dataset=dataset_name)
        else:
            print(f"使用文件路径方式加载数据集: {task_type}/{dataset_name}")
            return None
    except Exception as e:
        print(f"加载数据集 {dataset_name} 失败: {str(e)}")
        return None

def load_questions_from_dataset(task_type, dataset_name):
    """
    从指定任务类型和数据集名称中加载多模态问题
    
    Args:
        task_type: 任务类型，'cla', 'measurement', 'seg', 'report'中的一种
        dataset_name: 数据集名称
        
    Returns:
        包含多模态消息的问题列表
    """
    # 尝试使用数据集类加载
    dataset = load_task_dataset(task_type, dataset_name)
    
    if dataset is not None:
        # 使用数据集类加载数据
        try:
            questions = []
            for i in range(len(dataset)):
                line = dataset.data.iloc[i]
                msgs = dataset.build_prompt(line, task_type)
                
                # 提取图像路径
                image_paths = []
                text_content = ""
                for msg in msgs:
                    if msg["type"] == "image":
                        image_paths.append(msg["value"])
                    elif msg["type"] == "text":
                        text_content = msg["value"]
                
                question = {
                    'id': str(i),
                    'text_content': text_content,
                    'image_paths': image_paths,
                    'category': task_type
                }
                questions.append(question)
            return questions
        except Exception as e:
            print(f"从数据集类加载问题失败: {str(e)}")
    
    # 如果数据集类加载失败，尝试使用文件路径方式
    # 尝试不同的文件路径
    possible_paths = [
        # 直接使用相对路径
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", task_type, dataset_name, "questions.jsonl"),
        # 使用VLMEvalKit目录
        os.path.join("/media/ps/data-ssd/benchmark/VLMEvalKit/data", task_type, dataset_name, "questions.jsonl"),
        # 使用绝对路径
        os.path.join("/media/ps/data-ssd/benchmark/VLMEvalKit/vlmeval/data", task_type, dataset_name, "questions.jsonl"),
        # 使用用户目录
        os.path.join("/home/guohongcheng/data", task_type, dataset_name, "questions.jsonl")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"使用文件路径加载问题: {path}")
            try:
                questions = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        questions.append(json.loads(line.strip()))
                return questions
            except Exception as e:
                print(f"从文件加载问题失败: {str(e)}")
    
    # 如果所有路径都失败，尝试查找数据文件
    print(f"所有路径都失败，尝试查找数据文件...")
    try:
        # 使用find命令查找文件
        cmd = f"find /media/ps/data-ssd/benchmark -name 'questions.jsonl' | grep '{task_type}/{dataset_name}'"
        result = os.popen(cmd).read().strip()
        if result:
            path = result.split('\n')[0]
            print(f"找到数据文件: {path}")
            try:
                questions = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        questions.append(json.loads(line.strip()))
                return questions
            except Exception as e:
                print(f"从找到的文件加载问题失败: {str(e)}")
    except Exception as e:
        print(f"查找数据文件失败: {str(e)}")
    
    print(f"无法找到数据集 {task_type}/{dataset_name} 的问题文件")
    return []

def run_batch_test(task_type, dataset_name, output_dir, model_name=DEFAULT_MODEL_NAME, api_key=DEFAULT_API_KEY, sample_limit=None):
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
    print(f"开始批量测试: {task_type}/{dataset_name}, 模型: {model_name}")
    
    # 从数据集加载问题
    questions = load_questions_from_dataset(task_type, dataset_name)
    
    # 如果没有找到问题，返回
    if not questions:
        print(f"没有找到问题，终止测试")
        return
    
    # 限制样本数量
    if sample_limit and sample_limit > 0:
        questions = questions[:sample_limit]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 处理每个问题
    for i, question in enumerate(questions):
        print(f"处理问题 {i+1}/{len(questions)}: {question['id']}")
        
        # 构建消息
        messages = []
        
        # 添加图像
        for image_path in question["image_paths"]:
            messages.append({"type": "image", "value": image_path})
        
        # 添加文本
        messages.append({"type": "text", "value": question["text_content"]})
        
        # 调用模型
        response = call_model_multimodal(
            model_name=model_name,
            messages=messages,
            api_key=api_key
        )
        
        # 保存结果
        result = {
            "id": question["id"],
            "text_content": question["text_content"],
            "image_paths": question["image_paths"],
            "category": question.get("category", task_type),
            "response": response
        }
        
        save_question_result(
            result=result,
            model_name=model_name,
            questions_file_path=f"{task_type}/{dataset_name}",
            output_dir=output_dir,
            timestamp=timestamp
        )
    
    print(f"批量测试完成: {task_type}/{dataset_name}")

def get_available_datasets(task_type):
    """
    获取指定任务类型下的所有可用数据集
    
    Args:
        task_type: 任务类型，'cla', 'measurement', 'seg', 'report'中的一种
        
    Returns:
        该任务类型下可用的数据集名称列表
    """
    try:
        # 根据任务类型获取相应的数据集类的TSV_PATH字典的键作为可用数据集
        if task_type == 'cla':
            from vlmeval.dataset.image_cla import ClaDataset
            return list(ClaDataset.TSV_PATH.keys())
        elif task_type == 'measurement':
            from vlmeval.dataset.image_measurement import MeasureDataset
            return list(MeasureDataset.TSV_PATH.keys())
        elif task_type == 'seg':
            from vlmeval.dataset.image_seg import SegDataset
            return list(SegDataset.TSV_PATH.keys())
        elif task_type == 'report':
            from vlmeval.dataset.image_report import ReportDataset
            return list(ReportDataset.TSV_PATH.keys())
        else:
            print(f"不支持的任务类型: {task_type}")
            return []
    except Exception as e:
        print(f"获取可用数据集失败: {str(e)}")
        return []

def run_multi_task_test(task_types, output_dir, model_name=DEFAULT_MODEL_NAME, api_key=DEFAULT_API_KEY, test_all_datasets=False, sample_limit=None):
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
    # 数据集映射
    dataset_mapping = {
        "cla": ["anatomy", "plane"],
        "measurement": ["18", "27", "31", "50", "57"],
        "seg": ["10"],
        "report": ["10", "39"]
    }
    
    # 遍历任务类型
    for task_type in task_types:
        # 获取数据集列表
        if test_all_datasets:
            # 获取所有可用的数据集
            datasets = get_available_datasets(task_type)
            if not datasets:
                print(f"任务类型 {task_type} 没有找到可用的数据集")
                continue
            print(f"任务类型 {task_type} 找到 {len(datasets)} 个数据集")
        else:
            # 使用预定义的数据集
            datasets = dataset_mapping.get(task_type, [])
        
        # 遍历数据集
        for dataset_name in datasets:
            # 构建输出目录
            task_output_dir = os.path.join(output_dir, task_type, dataset_name)
            
            # 运行批量测试
            run_batch_test(
                task_type=task_type,
                dataset_name=dataset_name,
                output_dir=task_output_dir,
                model_name=model_name,
                api_key=api_key,
                sample_limit=sample_limit
            )

def main():
    parser = argparse.ArgumentParser(description='批量测试方舟平台API调用')
    parser.add_argument('--multi_task', action='store_true', help='是否使用多任务模式')
    parser.add_argument('--task_types', nargs='+', choices=['cla', 'measurement', 'seg', 'report'],
                        help='任务类型列表')
    parser.add_argument('--task_type', type=str, choices=['cla', 'measurement', 'seg', 'report'],
                        help='单个任务类型')
    parser.add_argument('--dataset_name', type=str, help='数据集名称，与task_type一起使用')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--test_all_datasets', action='store_true', help='是否测试所有数据集')
    parser.add_argument('--sample_limit', type=int, help='限制每个数据集的样本数量')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_NAME, help='模型名称')
    parser.add_argument('--api_key', type=str, default=DEFAULT_API_KEY, help='API密钥')

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
    elif args.multi_task and args.task_types:
        run_multi_task_test(
            task_types=args.task_types,
            output_dir=args.output_dir,
            model_name=args.model,
            api_key=args.api_key,
            test_all_datasets=args.test_all_datasets,
            sample_limit=args.sample_limit
        )
    else:
        print("错误: 必须指定task_type和dataset_name，或者指定--multi_task和task_types")
        parser.print_help()

if __name__ == "__main__":
    main()
