#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量测试脚本 - Gemini 2.5 Pro Preview专用版本
基于batch_test_xiaohu.py修改
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

# API配置
API_BASE = "https://xiaohumini.site"
API_RETRIES = 3
EMPTY_RESPONSE_RETRIES = 3
MIN_VALID_RESPONSE_LENGTH = 0

# Gemini模型配置
MODELS = {
    "gemini-2.5-pro-preview-03-25": {
        "name": "gemini-2.5-pro-preview-03-25",
        "api_key": "xxxx"
    }
}

# 修改为绝对导入
from vlmeval.dataset.image_cla import ClaDataset
from vlmeval.dataset.image_measurement import MeasureDataset 
from vlmeval.dataset.image_seg import SegDataset
from vlmeval.dataset.image_report import ReportDataset

# 添加默认API密钥字典
DEFAULT_API_KEYS = {
    'cla': "xxxx",
    'measurement': "xxxx",
    'seg': "xxx",
    'report': "xxxx"
}

# ... 保持其他函数不变 ...
# ... existing code ...

def get_available_datasets(task_type):
    """
    获取指定任务类型下的所有可用数据集
    
    Args:
        task_type: 任务类型 ('cla', 'measurement', 'seg', 'report')
        
    Returns:
        数据集名称列表
    """
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

def load_questions(file_path):
    """
    从文件加载问题
    支持.json和.jsonl格式
    
    Args:
        file_path: 问题文件路径
        
    Returns:
        问题列表
    """
    questions = []
    
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return questions
        
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
        else:
            print(f"错误: 不支持的文件格式: {file_path}")
    except Exception as e:
        print(f"加载问题文件时出错: {e}")
        
    return questions

def load_prompt(file_path):
    """
    加载提示词模板
    
    Args:
        file_path: 模板文件路径
        
    Returns:
        提示词模板字符串
    """
    if not os.path.exists(file_path):
        print(f"错误: 提示词模板文件不存在: {file_path}")
        return ""
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"加载提示词模板时出错: {e}")
        return ""

def load_questions_from_dataset(task_type, dataset_name):
    """
    从数据集加载多模态问题
    
    Args:
        task_type: 任务类型
        dataset_name: 数据集名称
        
    Returns:
        问题列表
    """
    try:
        if task_type == 'cla':
            dataset = ClaDataset(dataset=dataset_name)
        elif task_type == 'measurement':
            dataset = MeasureDataset(dataset=dataset_name)
        elif task_type == 'seg':
            dataset = SegDataset(dataset=dataset_name)
        elif task_type == 'report':
            dataset = ReportDataset(dataset=dataset_name)
        else:
            print(f"不支持的任务类型: {task_type}")
            return []
            
        return dataset.data
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return []

async def make_api_request(session, url, headers, data, retries=API_RETRIES):
    """
    发送API请求并处理重试
    
    Args:
        session: aiohttp会话
        url: API端点URL
        headers: 请求头
        data: 请求数据
        retries: 最大重试次数
        
    Returns:
        响应JSON或None
    """
    for i in range(retries):
        try:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"API请求失败 (尝试 {i+1}/{retries}): HTTP {response.status}")
                    if i < retries - 1:
                        await asyncio.sleep(1)  # 重试前等待
        except Exception as e:
            print(f"API请求出错 (尝试 {i+1}/{retries}): {e}")
            if i < retries - 1:
                await asyncio.sleep(1)
    return None

async def process_question(session, question, model_name, max_tokens, temperature):
    """
    处理单个问题
    
    Args:
        session: aiohttp会话
        question: 问题数据
        model_name: 模型名称
        max_tokens: 最大生成token数
        temperature: 温度参数
        
    Returns:
        处理结果字典
    """
    url = f"{API_BASE}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MODELS[model_name]['api_key']}"
    }
    
    data = {
        "model": model_name,
        "messages": question["messages"] if isinstance(question, dict) and "messages" in question else [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    start_time = time.time()
    response = await make_api_request(session, url, headers, data)
    end_time = time.time()
    
    if not response:
        return {
            "success": False,
            "error": "API请求失败",
            "time_taken": end_time - start_time
        }
        
    return {
        "success": True,
        "response": response,
        "time_taken": end_time - start_time
    }

async def run_batch_test_async(questions, model_name, max_tokens, temperature, 
                              save_results=True, output_dir='results',
                              latest_dir=None, archive_dir=None,
                              max_concurrent_requests=5,
                              questions_file_path=None):
    """
    异步运行批量测试
    
    Args:
        questions: 问题列表
        model_name: 模型名称
        max_tokens: 最大生成token数
        temperature: 温度参数
        save_results: 是否保存结果
        output_dir: 输出目录
        latest_dir: 最新结果目录
        archive_dir: 历史结果目录
        max_concurrent_requests: 最大并发请求数
        questions_file_path: 问题文件路径
        
    Returns:
        测试结果列表
    """
    results = []
    
    # 创建输出目录
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        if latest_dir:
            os.makedirs(latest_dir, exist_ok=True)
        if archive_dir:
            os.makedirs(archive_dir, exist_ok=True)
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 准备结果文件路径
    if save_results:
        results_file = os.path.join(model_output_dir, f'test_results_{timestamp}.jsonl')
        if latest_dir:
            latest_results_file = os.path.join(latest_dir, f'{model_name}_latest.jsonl')
        if archive_dir:
            archive_results_file = os.path.join(archive_dir, f'{model_name}_{timestamp}.jsonl')
    
    # 创建信号量控制并发
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async with aiohttp.ClientSession() as session:
        # 创建任务列表
        tasks = []
        for i, question in enumerate(questions):
            task = asyncio.create_task(
                process_question_with_semaphore(
                    semaphore, session, question, model_name,
                    max_tokens, temperature
                )
            )
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks)
    
    # 保存结果
    if save_results:
        # 保存到主结果文件
        with open(results_file, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        # 保存到最新结果文件
        if latest_dir:
            with open(latest_results_file, 'w', encoding='utf-8') as f:
                for result in results:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
        
        # 保存到历史结果文件
        if archive_dir:
            with open(archive_results_file, 'w', encoding='utf-8') as f:
                for result in results:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
    
    return results

async def process_question_with_semaphore(semaphore, session, question, model_name, max_tokens, temperature):
    """
    使用信号量控制并发的问题处理函数
    
    Args:
        semaphore: 信号量
        session: aiohttp会话
        question: 问题数据
        model_name: 模型名称
        max_tokens: 最大生成token数
        temperature: 温度参数
        
    Returns:
        处理结果
    """
    async with semaphore:
        return await process_question(session, question, model_name, max_tokens, temperature)

def run_batch_test(questions, prompt_template, model_name,
                   max_tokens=800, temperature=0.7,
                   save_results=True, output_dir='results',
                   latest_dir=None, archive_dir=None,
                   questions_file_path=None):
    """
    运行批量测试（同步版本）
    
    Args:
        questions: 问题列表
        prompt_template: 提示词模板
        model_name: 模型名称
        max_tokens: 最大生成token数
        temperature: 温度参数
        save_results: 是否保存结果
        output_dir: 输出目录
        latest_dir: 最新结果目录
        archive_dir: 历史结果目录
        questions_file_path: 问题文件路径
        
    Returns:
        测试结果列表
    """
    results = []
    
    # 创建输出目录
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        if latest_dir:
            os.makedirs(latest_dir, exist_ok=True)
        if archive_dir:
            os.makedirs(archive_dir, exist_ok=True)
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 准备结果文件路径
    if save_results:
        results_file = os.path.join(model_output_dir, f'test_results_{timestamp}.jsonl')
        if latest_dir:
            latest_results_file = os.path.join(latest_dir, f'{model_name}_latest.jsonl')
        if archive_dir:
            archive_results_file = os.path.join(archive_dir, f'{model_name}_{timestamp}.jsonl')
    
    print(f"开始测试模型 {model_name}，共 {len(questions)} 个问题")
    start_time = time.time()
    
    for i, question in enumerate(questions):
        print(f"处理问题 {i+1}/{len(questions)}")
        
        # 准备API请求
        url = f"{API_BASE}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MODELS[model_name]['api_key']}"
        }
        
        # 准备请求数据
        messages = []
        if prompt_template:
            messages.append({
                "role": "system",
                "content": prompt_template
            })
        
        if isinstance(question, dict) and "messages" in question:
            messages.extend(question["messages"])
        else:
            messages.append({
                "role": "user",
                "content": question
            })
        
        data = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # 发送请求
        response = None
        error = None
        start_request_time = time.time()
        
        for retry in range(API_RETRIES):
            try:
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    break
                else:
                    print(f"API请求失败 (尝试 {retry+1}/{API_RETRIES}): HTTP {response.status_code}")
                    if retry < API_RETRIES - 1:
                        time.sleep(1)  # 重试前等待
            except Exception as e:
                error = str(e)
                print(f"API请求出错 (尝试 {retry+1}/{API_RETRIES}): {e}")
                if retry < API_RETRIES - 1:
                    time.sleep(1)
        
        end_request_time = time.time()
        
        # 处理结果
        result = {
            "question_id": i,
            "question": question,
            "model": model_name,
            "time_taken": end_request_time - start_request_time
        }
        
        if response and response.status_code == 200:
            result["success"] = True
            result["response"] = response.json()
        else:
            result["success"] = False
            result["error"] = error or f"HTTP {response.status_code if response else 'No Response'}"
        
        results.append(result)
        
        # 保存中间结果
        if save_results:
            with open(results_file, 'a', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
            
            if latest_dir:
                with open(latest_results_file, 'a', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
            
            if archive_dir:
                with open(archive_results_file, 'a', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
    
    total_time = time.time() - start_time
    print(f"测试完成，总用时: {total_time:.2f} 秒")
    
    return results

def run_batch_test_multimodal(questions, model_name, 
                             max_tokens=800, temperature=0.7,
                             save_results=True, output_dir='results',
                             latest_dir=None, archive_dir=None,
                             questions_file_path=None):
    """
    运行多模态批量测试
    
    Args:
        questions: 多模态问题列表
        model_name: 模型名称
        max_tokens: 最大生成token数
        temperature: 温度参数
        save_results: 是否保存结果
        output_dir: 输出目录
        latest_dir: 最新结果目录
        archive_dir: 历史结果目录
        questions_file_path: 问题文件路径
        
    Returns:
        测试结果列表
    """
    results = []
    
    # 创建输出目录
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        if latest_dir:
            os.makedirs(latest_dir, exist_ok=True)
        if archive_dir:
            os.makedirs(archive_dir, exist_ok=True)
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 准备结果文件路径
    if save_results:
        results_file = os.path.join(model_output_dir, f'multimodal_test_results_{timestamp}.jsonl')
        if latest_dir:
            latest_results_file = os.path.join(latest_dir, f'{model_name}_latest.jsonl')
        if archive_dir:
            archive_results_file = os.path.join(archive_dir, f'{model_name}_{timestamp}.jsonl')
    
    print(f"开始多模态测试模型 {model_name}，共 {len(questions)} 个问题")
    start_time = time.time()
    
    for i, question in enumerate(questions):
        print(f"处理多模态问题 {i+1}/{len(questions)}")
        
        # 准备API请求
        url = f"{API_BASE}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MODELS[model_name]['api_key']}"
        }
        
        # 准备请求数据
        data = {
            "model": model_name,
            "messages": question["messages"] if isinstance(question, dict) and "messages" in question else [{"role": "user", "content": question}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # 发送请求
        response = None
        error = None
        start_request_time = time.time()
        
        for retry in range(API_RETRIES):
            try:
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    break
                else:
                    print(f"API请求失败 (尝试 {retry+1}/{API_RETRIES}): HTTP {response.status_code}")
                    if retry < API_RETRIES - 1:
                        time.sleep(1)  # 重试前等待
            except Exception as e:
                error = str(e)
                print(f"API请求出错 (尝试 {retry+1}/{API_RETRIES}): {e}")
                if retry < API_RETRIES - 1:
                    time.sleep(1)
        
        end_request_time = time.time()
        
        # 处理结果
        result = {
            "question_id": i,
            "question": question,
            "model": model_name,
            "time_taken": end_request_time - start_request_time
        }
        
        if response and response.status_code == 200:
            result["success"] = True
            result["response"] = response.json()
        else:
            result["success"] = False
            result["error"] = error or f"HTTP {response.status_code if response else 'No Response'}"
        
        results.append(result)
        
        # 保存中间结果
        if save_results:
            with open(results_file, 'a', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
            
            if latest_dir:
                with open(latest_results_file, 'a', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
            
            if archive_dir:
                with open(archive_results_file, 'a', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
    
    total_time = time.time() - start_time
    print(f"多模态测试完成，总用时: {total_time:.2f} 秒")
    
    return results

async def async_run_batch_test_multimodal(questions, model_name,
                                         max_tokens=800, temperature=0.7,
                                         save_results=True, output_dir='results',
                                         latest_dir=None, archive_dir=None,
                                         max_concurrent_requests=5,
                                         questions_file_path=None):
    """
    异步运行多模态批量测试
    
    Args:
        questions: 多模态问题列表
        model_name: 模型名称
        max_tokens: 最大生成token数
        temperature: 温度参数
        save_results: 是否保存结果
        output_dir: 输出目录
        latest_dir: 最新结果目录
        archive_dir: 历史结果目录
        max_concurrent_requests: 最大并发请求数
        questions_file_path: 问题文件路径
        
    Returns:
        测试结果列表
    """
    return await run_batch_test_async(
        questions=questions,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        save_results=save_results,
        output_dir=output_dir,
        latest_dir=latest_dir,
        archive_dir=archive_dir,
        max_concurrent_requests=max_concurrent_requests,
        questions_file_path=questions_file_path
    )

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='批量测试Gemini 2.5 Pro Preview')
    
    parser.add_argument('--questions', type=str, default='test_questions.jsonl',
                        help='测试问题文件路径 (.json或.jsonl格式)')
    parser.add_argument('--prompt', type=str, default='assistant_prompt.txt',
                        help='提示词模板文件路径')
    parser.add_argument('--max_tokens', type=int, default=800,
                        help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='温度参数，控制随机性')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='结果输出目录')
    parser.add_argument('--async_mode', action='store_true',
                        help='是否使用异步模式发送API请求')
    parser.add_argument('--max_concurrent', type=int, default=5,
                        help='异步模式下的最大并发请求数')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制测试的问题数量')
    parser.add_argument('--latest_dir', type=str, default=None,
                        help='最新结果目录（覆盖式）')
    parser.add_argument('--archive_dir', type=str, default=None,
                        help='历史结果目录（累积式）')
    parser.add_argument('--multimodal', action='store_true',
                        help='是否使用多模态模式（直接从数据集加载多模态问题）')
    parser.add_argument('--task_type', type=str, choices=['cla', 'measurement', 'seg', 'report'], default=None,
                        help='多模态任务类型，与multimodal参数一起使用')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='数据集名称，与multimodal参数一起使用')
    parser.add_argument('--multi_task', action='store_true',
                        help='是否使用多任务并行模式')
    parser.add_argument('--task_types', type=str, nargs='+', 
                        choices=['cla', 'measurement', 'seg', 'report'], default=None,
                        help='多任务模式下的任务类型列表')
    parser.add_argument('--dataset_names', type=str, nargs='+', default=None,
                        help='多任务模式下的数据集名称列表')
    parser.add_argument('--test_all_datasets', action='store_true',
                        help='是否测试每个任务类型下的所有可用数据集')
    parser.add_argument('--sample_limit', type=int, default=None,
                        help='限制每个数据集测试的样本数量')
    
    args = parser.parse_args()
    
    # 固定使用Gemini 2.5 Pro Preview模型
    model_name = "gemini-2.5-pro-preview-03-25"
    models_to_test = [model_name]
    
    # 传递问题文件路径给测试函数
    questions_file_path = args.questions
    
    # 检查多任务并行模式
    if args.multi_task:
        if not args.task_types:
            print("错误: 多任务并行模式需要指定task_types参数")
            return
        
        if not args.test_all_datasets and (not args.dataset_names or len(args.task_types) != len(args.dataset_names)):
            print("错误: 未指定--test_all_datasets时，需要为每个任务类型提供对应的数据集名称")
            return
            
        # 运行多任务并行测试
        run_multi_task_parallel(
            models=models_to_test,
            task_types=args.task_types,
            dataset_names=args.dataset_names,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            save_results=True,
            output_dir=args.output_dir,
            latest_dir=args.latest_dir,
            archive_dir=args.archive_dir,
            max_concurrent_requests=args.max_concurrent,
            test_all_datasets=args.test_all_datasets,
            sample_limit=args.sample_limit
        )
        return
    
    # 加载问题
    if args.multimodal:
        if not args.task_type or not args.dataset_name:
            print("错误: 多模态模式需要指定task_type和dataset_name参数")
            return
        
        questions = load_questions_from_dataset(args.task_type, args.dataset_name)
        if not questions:
            print(f"错误: 无法从数据集 {args.dataset_name} 加载多模态问题")
            return
        
        questions_file_path = f"{args.task_type}_{args.dataset_name}"
        print(f"多模态模式: 从数据集 {args.dataset_name} 加载了 {len(questions)} 个问题")
    else:
        questions = load_questions(args.questions)
    
    # 限制问题数量
    if args.limit and 0 < args.limit < len(questions):
        questions = questions[:args.limit]
        print(f"限制测试问题数量为: {args.limit}")
    
    # 加载提示词模板（非多模态模式才需要）
    prompt_template = None if args.multimodal else load_prompt(args.prompt)
    
    print(f"测试模型: {model_name}")
    print(f"共 {len(questions)} 个测试问题")
    
    # 运行测试
    if args.multimodal:
        if args.async_mode:
            asyncio.run(async_run_batch_test_multimodal(
                questions=questions,
                model_name=model_name,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                save_results=True,
                output_dir=args.output_dir,
                latest_dir=args.latest_dir,
                archive_dir=args.archive_dir,
                max_concurrent_requests=args.max_concurrent,
                questions_file_path=questions_file_path
            ))
        else:
            run_batch_test_multimodal(
                questions=questions,
                model_name=model_name,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                save_results=True,
                output_dir=args.output_dir,
                latest_dir=args.latest_dir,
                archive_dir=args.archive_dir,
                questions_file_path=questions_file_path
            )
    else:
        if args.async_mode:
            asyncio.run(async_run_batch_test(
                questions=questions,
                prompt_template=prompt_template,
                model_name=model_name,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                output_dir=args.output_dir,
                latest_dir=args.latest_dir,
                archive_dir=args.archive_dir,
                max_concurrent_requests=args.max_concurrent,
                questions_file_path=questions_file_path
            ))
        else:
            run_batch_test(
                questions=questions,
                prompt_template=prompt_template,
                model_name=model_name,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                output_dir=args.output_dir,
                latest_dir=args.latest_dir,
                archive_dir=args.archive_dir,
                questions_file_path=questions_file_path
            )

if __name__ == "__main__":
    main() 
