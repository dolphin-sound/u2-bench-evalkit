#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量测试脚本
使用role_play_model_test_reinforced.py中的函数测试多个问题
"""
import json
import os
import time

def load_config(config_file=None):
    # 加载配置文件
    # 如果没有提供配置文件，则使用默认配置
    if config_file is None:
        return {'llmModels': []}
    
    # 否则从文件加载配置
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            import json
            return json.load(f)
    except Exception as e:
        print(f"加载配置文件失败: {str(e)}")
        return {'llmModels': []}

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

def get_model_api_name(model_name, model_config=None):
    # 根据模型名称获取API调用所需的模型名称和API密钥
    # 如果有模型配置，则从配置中查找API密钥
    if model_config and 'llmModels' in model_config:
        for model in model_config['llmModels']:
            if model['name'] == model_name:
                return model_name, model.get('api_key')
    
    # 如果没有找到配置或配置中没有该模型，则使用默认的API密钥
    if model_name in MODELS and 'api_key' in MODELS[model_name]:
        return model_name, MODELS[model_name]['api_key']
    
    # 如果都没有找到，则返回模型名称和None
    return model_name, None

EMPTY_RESPONSE_RETRIES = 3
MIN_VALID_RESPONSE_LENGTH = 0

# 默认模型配置
MODELS = {
    "gpt-4o-mini": {
        "name": "gpt-4o-mini",
        "api_key": "YOUR API KEY"
    },
    "gpt-4o-2024": {
        "name": "gpt-4o-2024-08-06",
        "api_key": "YOUR API KEY"
    },
    "gpt-4o-2024-08-06": {
        "name": "gpt-4o-2024-08-06",
        "api_key": "YOUR API KEY"
    },
    "gemini-2.0-pro": {
        "name": "gemini-2.0-pro-exp-02-05",
        "api_key": "YOUR API KEY"
    },
    "claude-3-7": {
        "name": "claude-3-7-sonnet-20250219",
        "api_key": "YOUR API KEY"
    },
    "claude-3-7-sonnet-20250219": {
        "name": "claude-3-7-sonnet-20250219",
        "api_key": "YOUR API KEY"
    },
    "qwen-max": {
        "name": "qwen-max-2025-01-25",
        "api_key": "YOUR API KEY"
    },
    "gemini-2.0-pro-exp-02-05": {
        "name": "gemini-2.0-pro-exp-02-05",
        "api_key": "YOUR API KEY"
    },
    "qwen-max-2025-01-25": {
        "name": "qwen-max-2025-01-25",
        "api_key": "YOUR API KEY"
    },
    "gemini-1.5-pro-latest": {
        "name": "gemini-1.5-pro-latest",
        "api_key": "YOUR API KEY"
    }
}

# 修改为绝对导入
from vlmeval.dataset.image_cla import ClaDataset
from vlmeval.dataset.image_measurement import MeasureDataset 
from vlmeval.dataset.image_seg import SegDataset
from vlmeval.dataset.image_report import ReportDataset

# 添加默认API密钥字典
DEFAULT_API_KEYS = {
    "YOUR API KEY"
}

def extract_models_from_config(config):
    """
    从配置对象中提取模型列表
    
    Args:
        config: 配置对象
        
    Returns:
        模型名称列表
    """
    models = []
    if config and "llmModels" in config and isinstance(config["llmModels"], list):
        for model_info in config["llmModels"]:
            if "name" in model_info:
                models.append(model_info["name"])
    
    if models:
        print(f"从配置中提取了 {len(models)} 个模型")
        for i, model in enumerate(models):
            print(f"  {i+1}. {model}")
    else:
        print("未从配置中找到任何模型")
    
    return models

def load_questions(file_path):
    """
    加载测试问题
    支持JSON和JSONL格式
    """
    questions = []
    
    if file_path.endswith('.json'):
        # 加载JSON格式
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'questions' in data:
                for q in data['questions']:
                    questions.append({
                        'id': q.get('id', ''),
                        'content': q.get('content', ''),
                        'category': q.get('category', '')
                    })
            else:
                print("警告: JSON文件中没有找到'questions'字段")
    
    elif file_path.endswith('.jsonl'):
        # 加载JSONL格式
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        q = json.loads(line)
                        questions.append({
                            'id': q.get('id', ''),
                            'content': q.get('memory_content', ''),
                            'category': q.get('category', '')
                        })
                    except json.JSONDecodeError as e:
                        print(f"解析JSONL行时出错: {e}")
    
    else:
        print(f"不支持的文件格式: {file_path}")
    
    return questions

def load_prompt(file_path):
    """
    加载提示词模板
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"加载提示词文件时出错: {e}")
        return "你是一个有用的AI助手。"

# 添加一个工具函数用于提取文件名
def get_result_filename(question, questions_file_path):
    """
    从问题和问题文件路径生成结果文件名
    
    Args:
        question: 问题对象
        questions_file_path: 问题文件路径
    
    Returns:
        生成的结果文件名（不含扩展名）
    """
    # 如果问题有ID，使用ID作为文件名
    if 'id' in question and question['id']:
        return f"{question['id']}"
    
    # 否则使用问题文件的基本名称
    if questions_file_path:
        base_name = os.path.basename(questions_file_path)
        return os.path.splitext(base_name)[0]
    
    # 最后的备选方案：使用时间戳
    return f"result_{int(time.time())}"

# 添加保存单个问题结果的函数
def save_question_result(result, model_name, questions_file_path, latest_dir=None, archive_dir=None):
    """
    保存单个问题的结果
    
    Args:
        result: 单个问题的结果对象
        model_name: 模型名称
        questions_file_path: 问题文件路径
        latest_dir: 最新结果目录（覆盖式）
        archive_dir: 历史结果目录（累积式）
    """
    # 如果两个目录都不存在，不执行保存
    if not latest_dir and not archive_dir:
        return
    
    # 生成文件名
    result_filename = get_result_filename(result, questions_file_path)
    
    # 保存最新结果（覆盖式）
    if latest_dir:
        # 创建模型特定的目录
        model_latest_dir = os.path.join(latest_dir, model_name)
        if not os.path.exists(model_latest_dir):
            os.makedirs(model_latest_dir, exist_ok=True)
        
        # 保存结果文件（覆盖）
        latest_file = os.path.join(model_latest_dir, f"{result_filename}.json")
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 保存历史结果（累积式）
    if archive_dir:
        # 创建模型特定的目录
        model_archive_dir = os.path.join(archive_dir, model_name)
        if not os.path.exists(model_archive_dir):
            os.makedirs(model_archive_dir, exist_ok=True)
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_file = os.path.join(model_archive_dir, f"{result_filename}_{timestamp}.json")
        with open(archive_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

def run_batch_test(questions, prompt_template, model_name, 
                  max_tokens=800, temperature=0.7, 
                  save_results=True, output_dir='results',
                  latest_dir=None, archive_dir=None,
                  model_config=None, questions_file_path=None):
    """
    运行批量测试
    
    Args:
        questions: 问题列表
        prompt_template: 提示词模板
        model_name: 模型名称
        max_tokens: 最大生成token数
        temperature: 温度参数
        save_results: 是否保存结果
        output_dir: 输出目录 (兼容旧版）
        latest_dir: 最新结果目录
        archive_dir: 历史结果目录
        model_config: 模型配置信息
        questions_file_path: 问题文件路径
        
    Returns:
        测试结果列表
    """
    results = []
    
    print(f"开始批量测试模型 {model_name}，共 {len(questions)} 个问题")
    
    # 创建输出目录 (兼容旧版)
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 创建模型特定的输出目录 (兼容旧版)
    model_output_dir = os.path.join(output_dir, model_name)
    if save_results and not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir, exist_ok=True)
    
    # 创建最新结果和历史结果目录
    if save_results and latest_dir and not os.path.exists(latest_dir):
        os.makedirs(latest_dir, exist_ok=True)
    if save_results and archive_dir and not os.path.exists(archive_dir):
        os.makedirs(archive_dir, exist_ok=True)
    
    start_time = time.time()
    
    for i, question in enumerate(questions):
        q_id = question.get('id', f'q{i+1}')
        q_content = question.get('content', '')
        q_category = question.get('category', '')
        
        if not q_content:
            print(f"跳过问题 {q_id}: 内容为空")
            continue
        
        print(f"\n[{i+1}/{len(questions)}] 测试模型 {model_name} 问题 {q_id} ({q_category})")
        
        # 替换提示词中的占位符
        system_prompt = prompt_template.replace('{{USER_QUESTION}}', q_content)
        
        try:
            # 调用模型API
            response = call_model(
                model_name=model_name,
                message=q_content,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                model_config=model_config
            )
            
            # 存储结果
            result = {
                'id': q_id,
                'question': q_content,
                'category': q_category,
                'response': response,
                'model': model_name,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            results.append(result)
            
            # 保存单个问题结果到最新和历史目录
            if save_results:
                save_question_result(
                    result, 
                    model_name, 
                    questions_file_path, 
                    latest_dir, 
                    archive_dir
                )
            
        except Exception as e:
            print(f"处理问题 {q_id} 时出错: {str(e)}")
    
    total_time = time.time() - start_time
    print(f"\n模型 {model_name} 批量测试完成，总用时: {total_time:.2f} 秒")
    
    # 保存结果到传统目录(兼容旧版)
    if save_results and results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(model_output_dir, f'test_results_{timestamp}.jsonl')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"模型 {model_name} 测试结果已保存至: {output_file}")
    
    return results

def run_multi_model_test(models, questions, prompt_template, 
                        max_tokens=800, temperature=0.7,
                        save_results=True, output_dir='results',
                        latest_dir=None, archive_dir=None,
                        model_config=None, questions_file_path=None):
    """
    运行多模型批量测试
    依次使用不同模型测试所有问题
    
    Args:
        models: 模型名称列表
        questions: 问题列表
        prompt_template: 提示词模板
        max_tokens: 最大生成token数
        temperature: 温度参数
        save_results: 是否保存结果
        output_dir: 输出目录
        latest_dir: 最新结果目录
        archive_dir: 历史结果目录
        model_config: 模型配置信息
        questions_file_path: 问题文件路径
    
    Returns:
        所有模型的测试结果字典 {model_name: results}
    """
    all_results = {}
    
    print(f"开始多模型批量测试，共 {len(models)} 个模型，{len(questions)} 个问题")
    
    start_time = time.time()
    
    for i, model_name in enumerate(models):
        print(f"\n===== 测试模型 {i+1}/{len(models)}: {model_name} =====")
        
        # 测试当前模型
        model_results = run_batch_test(
            questions=questions,
            prompt_template=prompt_template,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            save_results=save_results,
            output_dir=output_dir,
            latest_dir=latest_dir,
            archive_dir=archive_dir,
            model_config=model_config,
            questions_file_path=questions_file_path
        )
        
        all_results[model_name] = model_results
        
        # 模型之间添加延迟，避免API速率限制
        if i < len(models) - 1:
            wait_time = 5 + random.random() * 5  # 模型间等待更长
            print(f"\n等待 {wait_time:.2f} 秒后继续下一个模型...")
            time.sleep(wait_time)
    
    total_time = time.time() - start_time
    print(f"\n所有模型测试完成，总用时: {total_time:.2f} 秒")
    
    # 保存汇总结果
    if save_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(output_dir, f'summary_{timestamp}.json')
        
        summary = {
            'total_models': len(models),
            'total_questions': len(questions),
            'models_tested': models,
            'total_time': total_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"测试汇总信息已保存至: {summary_file}")
    
    return all_results

# 修改worker函数，添加结果保存路径参数
def parallel_test_worker(args):
    """
    并行测试工作函数
    
    Args:
        args: 包含所有测试所需参数的元组
              (model_name, questions, prompt_template, max_tokens, temperature,
               save_results, output_dir, latest_dir, archive_dir, model_config, questions_file_path)
              
    Returns:
        (model_name, results)元组
    """
    model_name, questions, prompt_template, max_tokens, temperature, save_results, output_dir, latest_dir, archive_dir, model_config, questions_file_path = args
    
    print(f"\n===== 并行测试模型: {model_name} =====")
    model_results = run_batch_test(
        questions=questions,
        prompt_template=prompt_template,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        save_results=save_results,
        output_dir=output_dir,
        latest_dir=latest_dir,
        archive_dir=archive_dir,
        model_config=model_config,
        questions_file_path=questions_file_path
    )
    
    return (model_name, model_results)

# 修改并行测试函数
def run_parallel_model_test(models, questions, prompt_template, 
                           max_tokens=800, temperature=0.7,
                           save_results=True, output_dir='results',
                           latest_dir=None, archive_dir=None,
                           model_config=None, max_workers=None,
                           questions_file_path=None):
    """
    并行运行多模型批量测试
    使用多进程同时测试多个模型
    
    Args:
        models: 模型名称列表
        questions: 问题列表
        prompt_template: 提示词模板
        max_tokens: 最大生成token数
        temperature: 温度参数
        save_results: 是否保存结果
        output_dir: 输出目录
        latest_dir: 最新结果目录
        archive_dir: 历史结果目录
        model_config: 模型配置信息
        max_workers: 最大并行工作进程数（若为None则使用可用CPU核心数）
        questions_file_path: 问题文件路径
    
    Returns:
        所有模型的测试结果字典 {model_name: results}
    """
    all_results = {}
    
    print(f"开始并行多模型批量测试，共 {len(models)} 个模型，{len(questions)} 个问题")
    
    start_time = time.time()
    
    # 确定最大并行数
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(models))
    else:
        max_workers = min(max_workers, len(models))
    
    print(f"将使用 {max_workers} 个并行进程进行测试")
    
    # 准备任务参数
    tasks = []
    for model_name in models:
        # 为每个模型创建参数元组
        task_args = (model_name, questions, prompt_template, max_tokens, 
                    temperature, save_results, output_dir, latest_dir, archive_dir, model_config, questions_file_path)
        tasks.append(task_args)
    
    # 创建进程池
    with multiprocessing.Pool(processes=max_workers) as pool:
        # 提交所有任务
        results = pool.map(parallel_test_worker, tasks)
        
        # 处理结果
        for model_name, model_results in results:
            all_results[model_name] = model_results
    
    total_time = time.time() - start_time
    print(f"\n所有模型并行测试完成，总用时: {total_time:.2f} 秒")
    
    # 保存汇总结果
    if save_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(output_dir, f'parallel_summary_{timestamp}.json')
        
        summary = {
            'total_models': len(models),
            'total_questions': len(questions),
            'models_tested': models,
            'total_time': total_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"并行测试汇总信息已保存至: {summary_file}")
    
    return all_results

# 添加异步调用模型的函数
async def async_call_model(session, model_name, message, system_prompt, max_tokens=800, temperature=0.7, model_config=None):
    """
    异步调用模型API获取响应
    
    Args:
        session: aiohttp会话
        model_name: 模型名称
        message: 用户消息
        system_prompt: 系统提示词
        max_tokens: 最大生成token数
        temperature: 温度参数
        model_config: 模型配置
        
    Returns:
        模型响应
    """
    # 获取API模型名称和API KEY
    api_model_name, api_key = get_model_api_name(model_name, model_config)
    
    if not api_key:
        return f"错误: 未找到模型 {model_name} 的API KEY"
    
    # 构建消息
    formatted_messages = []
    if system_prompt:
        formatted_messages.append({"role": "system", "content": system_prompt})
    formatted_messages.append({"role": "user", "content": message})
    
    # 准备请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 准备请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 准备请求体
    payload = {
        "model": api_model_name,
        "messages": formatted_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1,
        "stream": False
    }
    
    # API端点
    api_endpoint = f"{API_BASE}/v1/chat/completions"
    
    # 尝试发送请求并等待响应
    for attempt in range(API_RETRIES + 1):
        try:
            if attempt > 0:
                # 在重试之前等待
                wait_time = 0.5 * attempt + random.random() * 0.5
                await asyncio.sleep(wait_time)
            
            async with session.post(api_endpoint, headers=headers, json=payload, timeout=90) as response:
                if response.status == 200:
                    # 处理成功响应
                    resp_json = await response.json()
                    
                    if 'choices' in resp_json and resp_json['choices']:
                        model_reply = resp_json['choices'][0]['message']['content']
                        return model_reply
                    else:
                        return "API response format error, no response content found"
                elif response.status == 429:
                    # 处理速率限制错误
                    continue  # 重试
                else:
                    # 处理其他错误
                    if attempt < API_RETRIES:
                        continue  # 重试
                    try:
                        error_text = await response.text()
                        return f"API call failed: {response.status}, {error_text}"
                    except:
                        return f"API call failed: {response.status}"
                    
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < API_RETRIES:
                continue  # 重试
            return f"Error occurred while calling API: {str(e)}"
    
    return "Multiple API call attempts failed"

async def async_run_batch_test(questions, prompt_template, model_name, 
                             max_tokens=800, temperature=0.7, 
                             save_results=True, output_dir='results',
                             latest_dir=None, archive_dir=None,
                             model_config=None, max_concurrent_requests=5,
                             questions_file_path=None):
    """
    异步运行批量测试，不等待API响应就发出下一个请求
    
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
        model_config: 模型配置信息
        max_concurrent_requests: 最大并发请求数
        questions_file_path: 问题文件路径
        
    Returns:
        测试结果列表
    """
    results = []
    
    print(f"开始异步批量测试模型 {model_name}，共 {len(questions)} 个问题，最大并发请求数: {max_concurrent_requests}")
    
    # 创建输出目录 (兼容旧版)
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 创建模型特定的输出目录 (兼容旧版)
    model_output_dir = os.path.join(output_dir, model_name)
    if save_results and not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir, exist_ok=True)
    
    # 创建最新结果和历史结果目录
    if save_results and latest_dir and not os.path.exists(latest_dir):
        os.makedirs(latest_dir, exist_ok=True)
    if save_results and archive_dir and not os.path.exists(archive_dir):
        os.makedirs(archive_dir, exist_ok=True)
    
    start_time = time.time()
    
    # 用于控制并发请求数的信号量
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def process_question(i, question):
        """处理单个问题的异步函数"""
        async with semaphore:  # 控制并发请求数
            q_id = question.get('id', f'q{i+1}')
            q_content = question.get('content', '')
            q_category = question.get('category', '')
            
            if not q_content:
                print(f"跳过问题 {q_id}: 内容为空")
                return None
            
            print(f"\n[{i+1}/{len(questions)}] 异步测试模型 {model_name} 问题 {q_id} ({q_category})")
            
            # 替换提示词中的占位符
            system_prompt = prompt_template.replace('{{USER_QUESTION}}', q_content)
            
            try:
                # 异步调用模型API
                async with aiohttp.ClientSession() as session:
                    response = await async_call_model(
                        session=session,
                        model_name=model_name,
                        message=q_content,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        model_config=model_config
                    )
                
                # 存储结果
                result = {
                    'id': q_id,
                    'question': q_content,
                    'category': q_category,
                    'response': response,
                    'model': model_name,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # 保存单个问题结果到最新和历史目录
                if save_results:
                    save_question_result(
                        result, 
                        model_name, 
                        questions_file_path, 
                        latest_dir, 
                        archive_dir
                    )
                
                return result
                
            except Exception as e:
                print(f"处理问题 {q_id} 时出错: {str(e)}")
                return None
    
    # 创建所有问题的异步任务
    tasks = [process_question(i, question) for i, question in enumerate(questions)]
    
    # 运行所有任务并等待结果
    task_results = await asyncio.gather(*tasks)
    
    # 过滤掉None结果并添加到结果列表
    results = [result for result in task_results if result is not None]
    
    total_time = time.time() - start_time
    print(f"\n模型 {model_name} 异步批量测试完成，总用时: {total_time:.2f} 秒")
    
    # 保存结果到传统目录(兼容旧版)
    if save_results and results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(model_output_dir, f'async_test_results_{timestamp}.jsonl')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"模型 {model_name} 异步测试结果已保存至: {output_file}")
    
    return results

async def async_run_multi_model_test(models, questions, prompt_template, 
                                   max_tokens=800, temperature=0.7,
                                   save_results=True, output_dir='results',
                                   latest_dir=None, archive_dir=None,
                                   model_config=None, max_concurrent_requests=5,
                                   questions_file_path=None):
    """
    异步运行多模型批量测试
    
    Args:
        models: 模型名称列表
        questions: 问题列表
        prompt_template: 提示词模板
        max_tokens: 最大生成token数
        temperature: 温度参数
        save_results: 是否保存结果
        output_dir: 输出目录
        latest_dir: 最新结果目录
        archive_dir: 历史结果目录
        model_config: 模型配置信息
        max_concurrent_requests: 每个模型最大并发请求数
        questions_file_path: 问题文件路径
        
    Returns:
        所有模型的测试结果字典 {model_name: results}
    """
    all_results = {}
    
    print(f"开始异步多模型批量测试，共 {len(models)} 个模型，{len(questions)} 个问题")
    
    start_time = time.time()
    
    # 依次测试每个模型
    for i, model_name in enumerate(models):
        print(f"\n===== 异步测试模型 {i+1}/{len(models)}: {model_name} =====")
        
        # 异步测试当前模型
        model_results = await async_run_batch_test(
            questions=questions,
            prompt_template=prompt_template,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            save_results=save_results,
            output_dir=output_dir,
            latest_dir=latest_dir,
            archive_dir=archive_dir,
            model_config=model_config,
            max_concurrent_requests=max_concurrent_requests,
            questions_file_path=questions_file_path
        )
        
        all_results[model_name] = model_results
    
    total_time = time.time() - start_time
    print(f"\n所有模型异步测试完成，总用时: {total_time:.2f} 秒")
    
    # 保存汇总结果
    if save_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(output_dir, f'async_summary_{timestamp}.json')
        
        summary = {
            'total_models': len(models),
            'total_questions': len(questions),
            'models_tested': models,
            'total_time': total_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"异步测试汇总信息已保存至: {summary_file}")
    
    return all_results

# 添加一个组合多进程与异步的测试函数
def run_parallel_async_test(models, questions, prompt_template, 
                           max_tokens=800, temperature=0.7,
                           save_results=True, output_dir='results',
                           latest_dir=None, archive_dir=None,
                           model_config=None, max_workers=None, 
                           max_concurrent_requests=5,
                           questions_file_path=None):
    """
    结合多进程并行和异步API请求的批量测试函数
    
    Args:
        models: 模型名称列表
        questions: 问题列表
        prompt_template: 提示词模板
        max_tokens: 最大生成token数
        temperature: 温度参数
        save_results: 是否保存结果
        output_dir: 输出目录
        latest_dir: 最新结果目录
        archive_dir: 历史结果目录
        model_config: 模型配置信息
        max_workers: 最大并行工作进程数（若为None则使用可用CPU核心数）
        max_concurrent_requests: 每个模型的最大并发请求数
        questions_file_path: 问题文件路径
        
    Returns:
        所有模型的测试结果字典 {model_name: results}
    """
    all_results = {}
    
    print(f"开始多进程并行+异步API请求的批量测试，共 {len(models)} 个模型，{len(questions)} 个问题")
    
    start_time = time.time()
    
    # 确定最大并行进程数
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(models))
    else:
        max_workers = min(max_workers, len(models))
    
    print(f"将使用 {max_workers} 个并行进程，每个进程内最大并发请求数: {max_concurrent_requests}")
    
    # 准备任务参数
    tasks = []
    for model_name in models:
        # 为每个模型创建参数元组
        task_args = (model_name, questions, prompt_template, max_tokens, 
                    temperature, save_results, output_dir, latest_dir, archive_dir, model_config, max_concurrent_requests, questions_file_path)
        tasks.append(task_args)
    
    # 创建进程池
    with multiprocessing.Pool(processes=max_workers) as pool:
        # 提交所有任务
        results = pool.map(parallel_async_worker, tasks)
        
        # 处理结果
        for model_name, model_results in results:
            all_results[model_name] = model_results
    
    total_time = time.time() - start_time
    print(f"\n所有模型测试完成，总用时: {total_time:.2f} 秒")
    
    # 保存汇总结果
    if save_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(output_dir, f'parallel_async_summary_{timestamp}.json')
        
        summary = {
            'total_models': len(models),
            'total_questions': len(questions),
            'models_tested': models,
            'total_time': total_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"测试汇总信息已保存至: {summary_file}")
    
    return all_results

# 修复异步模式中的事件循环警告
def run_async_test(models, questions, prompt_template, 
                  max_tokens=800, temperature=0.7,
                  save_results=True, output_dir='results',
                  latest_dir=None, archive_dir=None,
                  model_config=None, max_concurrent_requests=5):
    """
    运行异步测试的入口函数
    
    Args:
        models: 模型名称列表
        questions: 问题列表
        prompt_template: 提示词模板
        max_tokens: 最大生成token数
        temperature: 温度参数
        save_results: 是否保存结果
        output_dir: 输出目录
        latest_dir: 最新结果目录
        archive_dir: 历史结果目录
        model_config: 模型配置信息
        max_concurrent_requests: 每个模型最大并发请求数
        
    Returns:
        所有模型的测试结果
    """
    # 设置新的事件循环，避免警告
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    
    # 如果只有一个模型，直接测试
    if len(models) == 1:
        results = loop.run_until_complete(
            async_run_batch_test(
                questions=questions,
                prompt_template=prompt_template,
                model_name=models[0],
                max_tokens=max_tokens,
                temperature=temperature,
                save_results=save_results,
                output_dir=output_dir,
                latest_dir=latest_dir,
                archive_dir=archive_dir,
                model_config=model_config,
                max_concurrent_requests=max_concurrent_requests
            )
        )
        return {models[0]: results}
    else:
        # 多模型测试
        return loop.run_until_complete(
            async_run_multi_model_test(
                models=models,
                questions=questions,
                prompt_template=prompt_template,
                max_tokens=max_tokens,
                temperature=temperature,
                save_results=save_results,
                output_dir=output_dir,
                latest_dir=latest_dir,
                archive_dir=archive_dir,
                model_config=model_config,
                max_concurrent_requests=max_concurrent_requests
            )
        )

# 添加处理多模态输入的函数
def call_model_multimodal(model_name, messages, max_tokens=800, temperature=0, retries=API_RETRIES, 
                         model_config=None, empty_retries=EMPTY_RESPONSE_RETRIES):
    """
    调用模型API获取多模态输入（图像+文本）的响应
    
    Args:
        model_name: 模型名称
        messages: 多模态消息列表，每个消息是包含type和value字段的字典
                 例如: [{"type": "image", "value": "图像路径"}, {"type": "text", "value": "文本内容"}]
        max_tokens: 最大生成token数
        temperature: 温度参数
        retries: API调用失败时的最大重试次数
        model_config: 模型配置
        empty_retries: 空回复的最大重试次数
        
    Returns:
        模型响应文本
    """
    # 测试模式检查
    if model_name == "test-mode":
        print(f"\n测试模式，生成模拟响应...")
        text_content = next((msg['value'] for msg in messages if msg['type'] == 'text'), "")
        image_paths = [msg['value'] for msg in messages if msg['type'] == 'image']
        print(f"用户文本: {text_content[:100]}...")
        print(f"图像数量: {len(image_paths)}")
        if image_paths:
            print(f"图像路径示例: {image_paths[0]}")
        test_reply = "(测试模式响应) 这是一个包含多模态输入的模拟响应。实际模型会基于图像和文本内容生成合适的回答。"
        print(f"生成的模拟响应: {test_reply[:100]}...")
        return test_reply
    
    # 获取API模型名称和API KEY
    api_model_name, api_key = get_model_api_name(model_name, model_config)
    
    if not api_key:
        return f"错误: 未找到模型 {model_name} 的API KEY"
    
    print(f"\n调用模型 {model_name} (API使用: {api_model_name})...")
    
    # 提取消息内容用于日志
    text_content = next((msg['value'] for msg in messages if msg['type'] == 'text'), "")
    image_paths = [msg['value'] for msg in messages if msg['type'] == 'image']
    print(f"用户文本: {text_content[:100]}...")
    print(f"图像数量: {len(image_paths)}")
    
    # 构建API消息格式
    content_list = []
    for msg in messages:
        if msg['type'] == 'text':
            content_list.append({
                "type": "text",
                "text": msg['value']
            })
        elif msg['type'] == 'image':
            try:
                with open(msg['value'], 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_data}"
                    }
                })
            except Exception as e:
                print(f"处理图像 {msg['value']} 失败: {str(e)}")
                content_list.append({
                    "type": "text",
                    "text": f"[图像加载失败: {msg['value']}]"
                })
    
    # 创建API格式的消息
    formatted_messages = [{
        "role": "user",
        "content": content_list
    }]
    
    # 准备请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 准备请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 准备请求体
    payload = {
        "model": api_model_name,
        "messages": formatted_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1,
        "stream": False
    }
    
    # API端点
    api_endpoint = f"{API_BASE}/v1/chat/completions"
    print(f"API请求URL: {api_endpoint}")
    print(f"请求包含 {len(formatted_messages)} 条消息")
    print(f"生成参数: temperature={payload['temperature']}, top_p={payload['top_p']}")
    
    # 初始等待时间
    current_wait_time = 0.5
    print(f"等待 {current_wait_time:.2f} 秒后发起API调用...")
    time.sleep(current_wait_time)
    
    # 空回复重试计数
    empty_attempt = 0
    
    # 尝试多次API调用
    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                print(f"重试API调用 #{attempt}...")
                wait_time = 0.5 * (attempt) + random.random() * 0.1
                print(f"等待 {wait_time:.2f} 秒后重试...")
                time.sleep(wait_time)
            
            # 发送API请求
            response = requests.post(
                api_endpoint, 
                headers=headers, 
                json=payload, 
                timeout=30
            )
            
            # 打印响应详情
            print(f"API响应状态码: {response.status_code}")
            
            # 解析响应
            if response.status_code == 200:
                try:
                    resp_json = response.json()
                    if 'choices' in resp_json and resp_json['choices']:
                        model_reply = resp_json['choices'][0]['message']['content']
                        print(f"\n成功接收模型响应，长度 {len(model_reply)}")
                        
                        # 检查空回复或极短回复
                        if not model_reply or len(model_reply.strip()) < MIN_VALID_RESPONSE_LENGTH:
                            empty_attempt += 1
                            print(f"⚠️ 收到空或极短响应(长度: {len(model_reply.strip())})")
                            
                            if empty_attempt < empty_retries:
                                new_temp = min(0.9, temperature + 0.2 * empty_attempt)
                                print(f"调整温度为 {new_temp} 重试空响应")
                                payload["temperature"] = new_temp
                                time.sleep(0.2 * empty_attempt)
                                continue
                            else:
                                print(f"已达到最大空响应重试次数 ({empty_retries})")
                                return f"[错误: 模型返回空响应，已重试 {empty_retries} 次]"
                        
                        print(f"模型响应: {model_reply[:200]}...")
                        return model_reply
                    else:
                        print("警告: API响应中未找到内容")
                        if attempt < retries:
                            continue
                        return "API响应格式错误，未找到响应内容"
                except Exception as e:
                    print(f"解析API响应时出错: {str(e)}")
                    if attempt < retries:
                        continue
                    return "解析模型响应失败"
            elif response.status_code == 429:
                print(f"⚠️ API速率限制错误 (429)")
                if attempt < retries:
                    print(f"等待 1.0 秒后重试...")
                    time.sleep(1.0)
                    continue
                return "API调用被速率限制，请稍后再试"
            else:
                print(f"API调用失败，状态码: {response.status_code}")
                if attempt < retries:
                    continue
                return f"API调用失败: {response.status_code}"
                
        except Exception as e:
            print(f"调用API时发生错误: {str(e)}")
            if attempt < retries:
                continue
            return f"调用API时发生错误，请稍后再试: {str(e)}"
    
    return "多次API调用尝试失败"

async def async_call_model_multimodal(session, model_name, messages, max_tokens=800, 
                                     temperature=0.7, model_config=None):
    """
    异步调用模型API获取多模态输入的响应
    
    Args:
        session: aiohttp会话
        model_name: 模型名称
        messages: 多模态消息列表，每个消息是包含type和value字段的字典
                例如: [{"type": "image", "value": "图像路径"}, {"type": "text", "value": "文本内容"}]
        max_tokens: 最大生成token数
        temperature: 温度参数
        model_config: 模型配置
        
    Returns:
        模型响应
    """
    # 获取API模型名称和API KEY
    api_model_name, api_key = get_model_api_name(model_name, model_config)
    
    if not api_key:
        return f"错误: 未找到模型 {model_name} 的API KEY"
    
    # 构建API消息格式
    content_list = []
    for msg in messages:
        if msg['type'] == 'text':
            content_list.append({
                "type": "text",
                "text": msg['value']
            })
        elif msg['type'] == 'image':
            try:
                with open(msg['value'], 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_data}"
                    }
                })
            except Exception as e:
                print(f"处理图像 {msg['value']} 失败: {str(e)}")
                content_list.append({
                    "type": "text",
                    "text": f"[图像加载失败: {msg['value']}]"
                })
    
    # 创建API格式的消息
    formatted_messages = [{
        "role": "user",
        "content": content_list
    }]
    
    # 准备请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 准备请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 准备请求体
    payload = {
        "model": api_model_name,
        "messages": formatted_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1,
        "stream": False
    }
    
    # API端点
    api_endpoint = f"{API_BASE}/v1/chat/completions"
    
    # 尝试发送请求并等待响应
    for attempt in range(API_RETRIES + 1):
        try:
            if attempt > 0:
                # 在重试之前等待
                wait_time = 0.5 * attempt + random.random() * 0.5
                await asyncio.sleep(wait_time)
            
            async with session.post(api_endpoint, headers=headers, json=payload, timeout=90) as response:
                if response.status == 200:
                    # 处理成功响应
                    resp_json = await response.json()
                    
                    if 'choices' in resp_json and resp_json['choices']:
                        model_reply = resp_json['choices'][0]['message']['content']
                        return model_reply
                    else:
                        return "API response format error, no response content found"
                elif response.status == 429:
                    # 处理速率限制错误
                    continue  # 重试
                else:
                    # 处理其他错误
                    if attempt < API_RETRIES:
                        continue  # 重试
                    try:
                        error_text = await response.text()
                        return f"API call failed: {response.status}, {error_text}"
                    except:
                        return f"API call failed: {response.status}"
                    
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < API_RETRIES:
                continue  # 重试
            return f"Error occurred while calling API: {str(e)}"
    
    return "Multiple API call attempts failed"

# 修改函数，支持加载多种任务类型的数据集
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
        if task_type == 'cla':
            return ClaDataset(dataset=dataset_name)
        elif task_type == 'measurement':
            return MeasureDataset(dataset=dataset_name)
        elif task_type == 'seg':
            return SegDataset(dataset=dataset_name)
        elif task_type == 'report':
            return ReportDataset(dataset=dataset_name)
        else:
            print(f"不支持的任务类型: {task_type}")
            return None
    except Exception as e:
        print(f"加载数据集 {dataset_name} 失败: {str(e)}")
        return None

# 添加从数据集生成多模态问题的函数
def load_questions_from_dataset(task_type, dataset_name):
    """
    从指定任务类型和数据集名称中加载多模态问题
    
    Args:
        task_type: 任务类型，'cla', 'measurement', 'seg', 'report'中的一种
        dataset_name: 数据集名称
        
    Returns:
        包含多模态消息的问题列表
    """
    dataset = load_task_dataset(task_type, dataset_name)
    if dataset is None:
        return []
    
    questions = []
    for i in range(len(dataset)):
        line = dataset.data.iloc[i]
        # 使用数据集的build_prompt方法构建多模态消息
        msgs = dataset.build_prompt(line, task_type)
        
        # 构建问题对象
        question = {
            'id': str(line.get('index', i)),
            'messages': msgs,  # 保存完整的多模态消息
            'category': task_type,
        }
        questions.append(question)
    
    print(f"从数据集 {dataset_name} 加载了 {len(questions)} 个多模态问题")
    return questions


# 添加API连接检查函数
def check_api_connection(model_name, api_key):
    """
    检查API连接是否正常
    
    Args:
        model_name: 模型名称
        api_key: API密钥
        
    Returns:
        bool: 连接是否正常
    """
    if not api_key:
        print(f"错误: 模型 {model_name} 没有有效的API密钥")
        return False
    
    print(f"检查模型 {model_name} 的API连接...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "Hello, this is a test message. Please respond with 'API connection successful'."
            }
        ],
        "max_tokens": 20,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"API连接成功: {response.status_code}")
            return True
        else:
            print(f"API连接失败: {response.status_code}, {response.text}")
            return False
    except Exception as e:
        print(f"API连接异常: {str(e)}")
        return False


# 定义一个新的批量测试函数，用于处理多模态输入
def run_batch_test_multimodal(questions, model_name, 
                             max_tokens=800, temperature=0.7, 
                             save_results=True, output_dir='results',
                             latest_dir=None, archive_dir=None,
                             model_config=None, questions_file_path=None):
    """
    运行批量测试，支持多模态输入
    
    Args:
        questions: 问题列表，每个问题包含多模态消息
        model_name: 模型名称
        max_tokens: 最大生成token数
        temperature: 温度参数
        save_results: 是否保存结果
        output_dir: 输出目录 (兼容旧版）
        latest_dir: 最新结果目录
        archive_dir: 历史结果目录
        model_config: 模型配置信息
        questions_file_path: 问题文件路径
        
    Returns:
        测试结果列表
    """
    results = []
    start_time = time.time()
    
    print(f"开始多模态批量测试，模型: {model_name}，共 {len(questions)} 个问题")
    
    # 获取API模型名称和API KEY
    api_model_name, api_key = get_model_api_name(model_name, model_config)
    
    # 检查API连接
    if not check_api_connection(api_model_name, api_key):
        print(f"错误: 模型 {model_name} 的API连接失败，跳过测试")
        # 处理每个问题
    for i, question in enumerate(questions):
        question_id = question.get('id', f"q{i}")
        print(f"处理问题 {i+1}/{len(questions)}: {question_id}")
        
        # 构建多模态消息
        messages = []
        
        # 添加用户消息（可能包含图像和文本）
        user_content = []
        for item in question.get('messages', []):
            if item['type'] == 'image':
                # 图像消息
                image_path = item['value']
                if os.path.exists(image_path):
                    with open(image_path, "rb") as image_file:
                        import base64
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                else:
                    print(f"警告: 图像文件不存在: {image_path}")
            elif item['type'] == 'text':
                # 文本消息
                user_content.append({
                    "type": "text",
                    "text": item['value']
                })
        
        # 添加用户消息
        if user_content:
            messages.append({"role": "user", "content": user_content})
        else:
            # 如果没有多模态内容，添加默认文本
            messages.append({"role": "user", "content": "请回答问题"})
        
        # 准备请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 准备请求体
        payload = {
            "model": api_model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1,
            "stream": False
        }
        
        # API端点
        api_endpoint = f"{API_BASE}/v1/chat/completions"
        
        # 记录开始时间
        question_start_time = time.time()
        response_text = None
        
        # 尝试发送请求并等待响应
        for attempt in range(API_RETRIES + 1):
            try:
                if attempt > 0:
                    wait_time = 0.5 * attempt + random.random() * 0.5
                    time.sleep(wait_time)
                
                response = requests.post(
                    api_endpoint, 
                    headers=headers, 
                    json=payload, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    resp_json = response.json()
                    if 'choices' in resp_json and resp_json['choices']:
                        response_text = resp_json['choices'][0]['message']['content']
                        break
                    else:
                        response_text = "API响应格式错误，未找到响应内容"
                elif response.status_code == 429:
                    print(f"API速率限制错误 (429)")
                    continue
                else:
                    if attempt < API_RETRIES:
                        continue
                    response_text = f"API调用失败: {response.status_code}"
                    
            except Exception as e:
                if attempt < API_RETRIES:
                    continue
                response_text = f"调用API时发生错误: {str(e)}"
        
        if response_text is None:
            response_text = "多次API调用尝试失败"
        
        # 计算耗时
        question_time = time.time() - question_start_time
        
        # 构建结果对象
        result = {
            'id': question_id,
            'question': question,
            'response': response_text,
            'time': question_time
        }
        
        results.append(result)
        
        # 保存结果
        if save_results:
            # 构建结果文件名
            result_filename = f"{question_id}_{model_name}.json"
            
            # 保存到输出目录
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                result_path = os.path.join(output_dir, result_filename)
                with open(result_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result, f, ensure_ascii=False, indent=2)
            
            # 保存到最新结果目录（覆盖式）
            if latest_dir:
                os.makedirs(latest_dir, exist_ok=True)
                latest_path = os.path.join(latest_dir, result_filename)
                with open(latest_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result, f, ensure_ascii=False, indent=2)
            
            # 保存到历史结果目录（累积式）
            if archive_dir:
                os.makedirs(archive_dir, exist_ok=True)
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                archive_filename = f"{question_id}_{model_name}_{timestamp}.json"
                archive_path = os.path.join(archive_dir, archive_filename)
                with open(archive_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 打印进度
        print(f"完成问题 {i+1}/{len(questions)}: 耗时 {question_time:.2f}秒")
    
    # 计算总耗时
    total_time = time.time() - start_time
    print(f"批量测试完成，共 {len(questions)} 个问题，总耗时: {total_time:.2f}秒")
    
    # 将所有结果合并为一个JSONL文件
    if save_results and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        combined_output_file = os.path.join(output_dir, f"{model_name}_combined_results.jsonl")
        with open(combined_output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"已将所有结果合并保存到: {combined_output_file}")
    
    return results
    
    if not api_key:
        return f"错误: 未找到模型 {model_name} 的API KEY"
    
    # 设置系统提示词
    system_prompt = None
    message = "请回答问题"
    
    # 构建消息
    formatted_messages = []
    if system_prompt:
        formatted_messages.append({"role": "system", "content": system_prompt})
    formatted_messages.append({"role": "user", "content": message})
    
    # 准备请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 准备请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 准备请求体
    payload = {
        "model": api_model_name,
        "messages": formatted_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1,
        "stream": False
    }
    
    # API端点
    api_endpoint = f"{API_BASE}/v1/chat/completions"
    
    # 尝试发送请求并等待响应
    for attempt in range(API_RETRIES + 1):
        try:
            if attempt > 0:
                wait_time = 0.5 * attempt + random.random() * 0.5
                time.sleep(wait_time)
            
            response = requests.post(
                api_endpoint, 
                headers=headers, 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                resp_json = response.json()
                if 'choices' in resp_json and resp_json['choices']:
                    return resp_json['choices'][0]['message']['content']
                else:
                    return "API响应格式错误，未找到响应内容"
            elif response.status_code == 429:
                print(f"API速率限制错误 (429)")
                continue
            else:
                if attempt < API_RETRIES:
                    continue
                return f"API调用失败: {response.status_code}"
                
        except Exception as e:
            if attempt < API_RETRIES:
                continue
            return f"调用API时发生错误: {str(e)}"
    
    return "多次API调用尝试失败"

# 修改main函数，添加测试所有数据集的选项和使用默认API密钥字典
def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='批量测试API调用')
    
    parser.add_argument('--questions', type=str, default='test_questions.jsonl',
                        help='测试问题文件路径 (.json或.jsonl格式)')
    parser.add_argument('--prompt', type=str, default='assistant_prompt.txt',
                        help='提示词模板文件路径')
    parser.add_argument('--model', type=str, default=None, action='append',
                        help='模型名称, 可多次使用此参数指定多个模型')
    parser.add_argument('--config', type=str, default='llm_config_silicon.json',
                        help='模型配置文件路径')
    parser.add_argument('--max_tokens', type=int, default=800,
                        help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='温度参数，控制随机性')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='结果输出目录')
    parser.add_argument('--model_index', type=int, default=None,
                        help='测试配置文件中指定索引的模型（从1开始）')
    parser.add_argument('--parallel', action='store_true',
                        help='是否使用并行模式测试多个模型')
    parser.add_argument('--async_mode', action='store_true',
                        help='是否使用异步模式发送API请求（不等待响应就发出下一个请求）')
    parser.add_argument('--parallel_async', action='store_true',
                        help='是否同时使用并行模式和异步模式（多进程并行处理模型，每个进程内使用异步API请求）')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='并行测试时的最大工作进程数')
    parser.add_argument('--max_concurrent', type=int, default=5,
                        help='异步模式下每个模型的最大并发请求数')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制测试的问题数量')
    
    # 添加新的命令行参数
    parser.add_argument('--latest_dir', type=str, default=None,
                        help='最新结果目录（覆盖式）')
    parser.add_argument('--archive_dir', type=str, default=None,
                        help='历史结果目录（累积式）')
    
    # 添加多模态相关参数
    parser.add_argument('--multimodal', action='store_true',
                        help='是否使用多模态模式（直接从数据集加载多模态问题）')
    parser.add_argument('--task_type', type=str, choices=['cla', 'measurement', 'seg', 'report'], default=None,
                        help='多模态任务类型，与multimodal参数一起使用')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='数据集名称，与multimodal参数一起使用')
    
    # 添加多任务并行相关参数 - API密钥变为可选
    parser.add_argument('--multi_task', action='store_true',
                        help='是否使用多任务并行模式（使用多个API key并行执行不同任务）')
    parser.add_argument('--task_types', type=str, nargs='+', 
                        choices=['cla', 'measurement', 'seg', 'report'], default=None,
                        help='多任务模式下的任务类型列表')
    parser.add_argument('--dataset_names', type=str, nargs='+', default=None,
                        help='多任务模式下的数据集名称列表')
    parser.add_argument('--api_keys', type=str, nargs='+', default=None,
                        help='可选：多任务模式下的API Key列表。若不提供，将使用内置API密钥')
    parser.add_argument('--test_all_datasets', action='store_true',
                        help='是否测试每个任务类型下的所有可用数据集')
    parser.add_argument('--sample_limit', type=int, default=None,
                        help='限制每个数据集测试的样本数量，设置为1表示只测试每个数据集的第一个样本')
    
    args = parser.parse_args()
    
    # 修复命令行参数，将--model转换为列表
    if args.model is None:
        args.model = []
    
    # 传递问题文件路径给测试函数，用于生成结果文件名
    questions_file_path = args.questions
    
    # 加载模型配置
    # 使用内置的模型配置
    model_config = {'llmModels': []}
    for model_name, model_info in MODELS.items():
        model_config['llmModels'].append({'name': model_name, 'api_key': model_info.get('api_key')})
    
    # 确定要测试的模型
    if args.model:
        # 使用命令行指定的模型列表
        models_to_test = args.model
    elif args.model_index is not None:
        # 使用指定索引的模型
        available_models = extract_models_from_config(model_config)
        if 1 <= args.model_index <= len(available_models):
            models_to_test = [available_models[args.model_index - 1]]
        else:
            print(f"模型索引超出范围: {args.model_index}，应在1到{len(available_models)}之间")
            return
    else:
        # 使用配置文件中的所有模型
        models_to_test = extract_models_from_config(model_config)
    
    # 检查多任务并行模式
    if args.multi_task:
        # 验证必要的参数
        if not args.task_types:
            print("错误: 多任务并行模式需要指定task_types参数")
            return
        
        # 如果测试所有数据集，则不需要dataset_names参数
        if not args.test_all_datasets and (not args.dataset_names or len(args.task_types) != len(args.dataset_names)):
            print("错误: 未指定--test_all_datasets时，需要为每个任务类型提供对应的数据集名称")
            return
        
        # API密钥检查变为可选
        if args.api_keys and len(args.task_types) != len(args.api_keys):
            print("警告: 提供的api_keys数量与task_types不匹配，将使用默认API密钥")
            args.api_keys = None
            
        # 运行多任务并行测试 - API密钥变为可选参数
        run_multi_task_parallel(
            models=models_to_test,
            task_types=args.task_types,
            dataset_names=args.dataset_names,
            api_keys=args.api_keys,  # 可能为None，函数内部会处理
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            save_results=True,
            output_dir=args.output_dir,
            latest_dir=args.latest_dir,
            archive_dir=args.archive_dir,
            model_config=model_config,
            max_concurrent_requests=args.max_concurrent,
            test_all_datasets=args.test_all_datasets,
            sample_limit=args.sample_limit
        )
        return
    
    # 以下是原有的单任务逻辑...
    # 加载问题
    if args.multimodal:
        # 检查必要的参数
        if not args.task_type or not args.dataset_name:
            print("错误: 多模态模式需要指定task_type和dataset_name参数")
            return
        
        # 从数据集加载多模态问题
        questions = load_questions_from_dataset(args.task_type, args.dataset_name)
        if not questions:
            print(f"错误: 无法从数据集 {args.dataset_name} 加载多模态问题")
            return
        
        # 设置文件路径用于生成结果文件名
        questions_file_path = f"{args.task_type}_{args.dataset_name}"
        
        print(f"多模态模式: 从数据集 {args.dataset_name} 加载了 {len(questions)} 个问题")
    else:
        # 从文件加载普通问题
        questions = load_questions(args.questions)
    
    # 限制问题数量
    if args.limit and 0 < args.limit < len(questions):
        questions = questions[:args.limit]
        print(f"限制测试问题数量为: {args.limit}")
    
    # 加载提示词模板（非多模态模式才需要）
    if not args.multimodal:
        prompt_template = load_prompt(args.prompt)
    else:
        prompt_template = None
    
    print(f"要测试的模型: {', '.join(models_to_test)}")
    print(f"共 {len(questions)} 个测试问题")
    
    # 运行测试
    if len(models_to_test) == 1:
        # 单个模型测试
        if args.multimodal:
            # 多模态测试
            if args.async_mode:
                # 多模态异步模式
                asyncio.run(async_run_batch_test_multimodal(
                    questions=questions,
                    model_name=models_to_test[0],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    save_results=True,
                    output_dir=args.output_dir,
                    latest_dir=args.latest_dir,
                    archive_dir=args.archive_dir,
                    model_config=model_config,
                    max_concurrent_requests=args.max_concurrent,
                    questions_file_path=questions_file_path
                ))
            else:
                # 多模态常规模式
                run_batch_test_multimodal(
                    questions=questions,
                    model_name=models_to_test[0],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    save_results=True,
                    output_dir=args.output_dir,
                    latest_dir=args.latest_dir,
                    archive_dir=args.archive_dir,
                    model_config=model_config,
                    questions_file_path=questions_file_path
                )
        else:
            # 普通测试（非多模态）
            if args.async_mode:
                # 使用异步模式
                asyncio.run(async_run_batch_test(
                    questions=questions,
                    prompt_template=prompt_template,
                    model_name=models_to_test[0],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    output_dir=args.output_dir,
                    latest_dir=args.latest_dir,
                    archive_dir=args.archive_dir,
                    model_config=model_config,
                    max_concurrent_requests=args.max_concurrent,
                    questions_file_path=questions_file_path
                ))
            else:
                # 使用常规模式
                run_batch_test(
                    questions=questions,
                    prompt_template=prompt_template,
                    model_name=models_to_test[0],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    output_dir=args.output_dir,
                    latest_dir=args.latest_dir,
                    archive_dir=args.archive_dir,
                    model_config=model_config,
                    questions_file_path=questions_file_path
                )
    else:
        # 多模型测试
        if args.multimodal:
            # 多模态多模型测试
            if args.parallel_async:
                # 多模态并行+异步模式
                run_parallel_async_test_multimodal(
                    models=models_to_test,
                    questions=questions,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    save_results=True,
                    output_dir=args.output_dir,
                    latest_dir=args.latest_dir,
                    archive_dir=args.archive_dir,
                    model_config=model_config,
                    max_workers=args.max_workers,
                    max_concurrent_requests=args.max_concurrent,
                    questions_file_path=questions_file_path
                )
            elif args.async_mode:
                # 多模态异步模式（串行多模型）
                # 为多模态版本异步多模型测试运行所有模型
                for model_name in models_to_test:
                    print(f"\n===== 异步测试多模态模型 {model_name} =====")
                    asyncio.run(async_run_batch_test_multimodal(
                        questions=questions,
                        model_name=model_name,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        save_results=True,
                        output_dir=args.output_dir,
                        latest_dir=args.latest_dir,
                        archive_dir=args.archive_dir,
                        model_config=model_config,
                        max_concurrent_requests=args.max_concurrent,
                        questions_file_path=questions_file_path
                    ))
            else:
                # 多模态常规模式（串行多模型）
                for model_name in models_to_test:
                    print(f"\n===== 测试多模态模型 {model_name} =====")
                    run_batch_test_multimodal(
                        questions=questions,
                        model_name=model_name,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        save_results=True,
                        output_dir=args.output_dir,
                        latest_dir=args.latest_dir,
                        archive_dir=args.archive_dir,
                        model_config=model_config,
                        questions_file_path=questions_file_path
                    )
        else:
            # 普通测试（非多模态多模型）
            # 根据是否启用并行模式选择不同的函数
            if args.parallel_async:
                # 并行+异步模式
                run_parallel_async_test(
                    models=models_to_test,
                    questions=questions,
                    prompt_template=prompt_template,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    save_results=True,
                    output_dir=args.output_dir,
                    latest_dir=args.latest_dir,
                    archive_dir=args.archive_dir,
                    model_config=model_config,
                    max_workers=args.max_workers,
                    max_concurrent_requests=args.max_concurrent,
                    questions_file_path=questions_file_path
                )
            elif args.parallel:
                # 仅并行模式
                run_parallel_model_test(
                    models=models_to_test,
                    questions=questions,
                    prompt_template=prompt_template,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    save_results=True,
                    output_dir=args.output_dir,
                    latest_dir=args.latest_dir,
                    archive_dir=args.archive_dir,
                    model_config=model_config,
                    max_workers=args.max_workers,
                    questions_file_path=questions_file_path
                )
            elif args.async_mode:
                # 仅异步模式
                asyncio.run(async_run_multi_model_test(
                    models=models_to_test,
                    questions=questions,
                    prompt_template=prompt_template,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    save_results=True,
                    output_dir=args.output_dir,
                    latest_dir=args.latest_dir,
                    archive_dir=args.archive_dir,
                    model_config=model_config,
                    max_concurrent_requests=args.max_concurrent,
                    questions_file_path=questions_file_path
                ))
            else:
                # 常规串行模式
                run_multi_model_test(
                    models=models_to_test,
                    questions=questions,
                    prompt_template=prompt_template,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    save_results=True,
                    output_dir=args.output_dir,
                    latest_dir=args.latest_dir,
                    archive_dir=args.archive_dir,
                    model_config=model_config,
                    questions_file_path=questions_file_path
                )

if __name__ == "__main__":
    main() 