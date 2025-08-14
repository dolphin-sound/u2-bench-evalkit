#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量测试脚本
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

# API配置
API_BASE = "https://xiaohumini.site"
API_RETRIES = 3
EMPTY_RESPONSE_RETRIES = 3
MIN_VALID_RESPONSE_LENGTH = 0

# 添加默认API密钥字典
DEFAULT_API_KEYS = {
    "YOUR API KEY"
}

# 只保留目标模型配置
MODELS = {
    "gemini-2.5-pro-preview-03-25": {
        "name": "gemini-2.5-pro-preview-03-25",
        "api_key": "YOUR API KEY"
    }
}

# 修改为绝对导入
from vlmeval.dataset.image_cla import ClaDataset
from vlmeval.dataset.image_measurement import MeasureDataset 
from vlmeval.dataset.image_seg import SegDataset
from vlmeval.dataset.image_report import ReportDataset

def extract_models_from_config(config):
    """
    从配置对象中提取模型列表，但只返回gemini-2.5-pro-preview-03-25
    
    Args:
        config: 配置对象
        
    Returns:
        模型名称列表，只包含gemini-2.5-pro-preview-03-25
    """
    models = ["gemini-2.5-pro-preview-03-25"]
    print("只使用模型: gemini-2.5-pro-preview-03-25")
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
    
    # 打印发送给API的消息
    print("\n===== 发送给API的多模态消息 =====")
    # 为了便于阅读，将base64图像数据替换为占位符
    formatted_messages_for_print = json.loads(json.dumps(formatted_messages))
    for msg in formatted_messages_for_print:
        if isinstance(msg.get('content'), list):
            for content in msg['content']:
                if content.get('type') == 'image_url':
                    content['image_url']['url'] = '[BASE64_IMAGE_DATA]'
    print(json.dumps(formatted_messages_for_print, ensure_ascii=False, indent=2))
    print("="*30)
    
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
    for attempt in range(retries + 1):
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
                print(f"API速率限制错误 (429)")
                continue
            else:
                if attempt < retries:
                    continue
                return f"API调用失败: {response.status_code}"
                
        except Exception as e:
            if attempt < retries:
                continue
            return f"调用API时发生错误: {str(e)}"
    
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
    
    print(f"开始批量测试模型 {model_name}，共 {len(questions)} 个多模态问题")
    
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
        q_messages = question.get('messages', [])
        q_category = question.get('category', '')
        
        if not q_messages:
            print(f"跳过问题 {q_id}: 消息为空")
            continue
        
        print(f"\n[{i+1}/{len(questions)}] 测试模型 {model_name} 问题 {q_id} ({q_category})")
        
        try:
            # 调用多模态模型API
            response = call_model_multimodal(
                model_name=model_name,
                messages=q_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                model_config=model_config
            )
            
            # 获取文本内容用于记录
            text_content = next((msg['value'] for msg in q_messages if msg['type'] == 'text'), "")
            image_paths = [msg['value'] for msg in q_messages if msg['type'] == 'image']
            
            # 存储结果
            result = {
                'id': q_id,
                'text_content': text_content,
                'image_paths': image_paths,
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
        output_file = os.path.join(model_output_dir, f'multimodal_test_results_{timestamp}.jsonl')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"模型 {model_name} 多模态测试结果已保存至: {output_file}")
    
    return results

# 添加异步版本的多模态批量测试函数
async def async_run_batch_test_multimodal(questions, model_name, 
                                        max_tokens=800, temperature=0.7, 
                                        save_results=True, output_dir='results',
                                        latest_dir=None, archive_dir=None,
                                        model_config=None, max_concurrent_requests=5,
                                        questions_file_path=None):
    """
    异步运行批量测试，支持多模态输入
    
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
        max_concurrent_requests: 最大并发请求数
        questions_file_path: 问题文件路径
        
    Returns:
        测试结果列表
    """
    results = []
    
    print(f"开始异步批量测试模型 {model_name}，共 {len(questions)} 个多模态问题，最大并发请求数: {max_concurrent_requests}")
    
    # 创建输出目录
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    model_output_dir = os.path.join(output_dir, model_name)
    if save_results and not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir, exist_ok=True)
    
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
            q_messages = question.get('messages', [])
            q_category = question.get('category', '')
            
            if not q_messages:
                print(f"跳过问题 {q_id}: 消息为空")
                return None
            
            print(f"\n[{i+1}/{len(questions)}] 异步测试模型 {model_name} 问题 {q_id} ({q_category})")
            
            try:
                # 异步调用多模态模型API
                async with aiohttp.ClientSession() as session:
                    response = await async_call_model_multimodal(
                        session=session,
                        model_name=model_name,
                        messages=q_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        model_config=model_config
                    )
                
                # 获取文本内容用于记录
                text_content = next((msg['value'] for msg in q_messages if msg['type'] == 'text'), "")
                image_paths = [msg['value'] for msg in q_messages if msg['type'] == 'image']
                
                # 存储结果
                result = {
                    'id': q_id,
                    'text_content': text_content,
                    'image_paths': image_paths,
                    'category': q_category,
                    'response': response,
                    'model': model_name,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # 保存单个问题结果
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
    print(f"\n模型 {model_name} 异步多模态批量测试完成，总用时: {total_time:.2f} 秒")
    
    # 保存结果到传统目录
    if save_results and results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(model_output_dir, f'async_multimodal_test_results_{timestamp}.jsonl')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"模型 {model_name} 异步多模态测试结果已保存至: {output_file}")
    
    return results

# 在main函数之前添加这两个新函数

def parallel_async_worker_multimodal(args):
    """
    多模态版本的并行异步工作函数
    
    Args:
        args: 包含所有测试所需参数的元组
              (model_name, questions, max_tokens, temperature,
               save_results, output_dir, latest_dir, archive_dir, model_config, 
               max_concurrent_requests, questions_file_path)
              
    Returns:
        (model_name, results)元组
    """
    model_name, questions, max_tokens, temperature, save_results, output_dir, latest_dir, archive_dir, model_config, max_concurrent_requests, questions_file_path = args
    
    print(f"\n===== 并行异步测试多模态模型: {model_name} =====")
    
    # 设置新的事件循环
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    
    # 运行异步函数
    model_results = loop.run_until_complete(
        async_run_batch_test_multimodal(
            questions=questions,
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
    )
    
    return (model_name, model_results)

def run_parallel_async_test_multimodal(models, questions, max_tokens=800, temperature=0.7,
                                      save_results=True, output_dir='results',
                                      latest_dir=None, archive_dir=None,
                                      model_config=None, max_workers=None, 
                                      max_concurrent_requests=5, questions_file_path=None):
    """
    多模态版本的并行异步测试函数，结合多进程并行和异步API请求
    
    Args:
        models: 模型名称列表
        questions: 多模态问题列表
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
    
    print(f"开始多进程并行+异步API请求的多模态批量测试，共 {len(models)} 个模型，{len(questions)} 个问题")
    
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
        task_args = (model_name, questions, max_tokens, 
                    temperature, save_results, output_dir, latest_dir, archive_dir, model_config, 
                    max_concurrent_requests, questions_file_path)
        tasks.append(task_args)
    
    # 创建进程池
    with multiprocessing.Pool(processes=max_workers) as pool:
        # 提交所有任务
        results = pool.map(parallel_async_worker_multimodal, tasks)
        
        # 处理结果
        for model_name, model_results in results:
            all_results[model_name] = model_results
    
    total_time = time.time() - start_time
    print(f"\n所有多模态模型并行异步测试完成，总用时: {total_time:.2f} 秒")
    
    # 保存汇总结果
    if save_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(output_dir, f'multimodal_parallel_async_summary_{timestamp}.json')
        
        summary = {
            'total_models': len(models),
            'total_questions': len(questions),
            'models_tested': models,
            'total_time': total_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"多模态并行异步测试汇总信息已保存至: {summary_file}")
    
    return all_results

# 添加一个函数获取任务类型下的所有数据集
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
        print(f"获取任务 {task_type} 的可用数据集时出错: {str(e)}")
        return []

# 添加一个新函数用于并行运行多任务模型测试
def run_multi_task_parallel(models, task_types, dataset_names=None, api_keys=None,
                           max_tokens=800, temperature=0.7,
                           save_results=True, output_dir='results',
                           latest_dir=None, archive_dir=None,
                           model_config=None, max_concurrent_requests=5,
                           test_all_datasets=False, sample_limit=None):
    """
    使用多个API Key并行运行多任务测试
    
    Args:
        models: 模型名称列表
        task_types: 任务类型列表，应包含 'cla', 'measurement', 'seg', 'report'
        dataset_names: 数据集名称列表，与task_types对应。若为None且test_all_datasets为True，则自动测试每个任务的所有数据集
        api_keys: API Key列表，与task_types对应，每个任务使用不同的API Key (可选，如不提供则使用内置的API密钥)
        max_tokens: 最大生成token数
        temperature: 温度参数
        save_results: 是否保存结果
        output_dir: 输出目录
        latest_dir: 最新结果目录
        archive_dir: 历史结果目录
        model_config: 模型配置信息
        max_concurrent_requests: 每个模型的最大并发请求数
        test_all_datasets: 是否测试每个任务类型下的所有数据集
        sample_limit: 限制每个数据集测试的样本数量。设置为1表示只测试每个数据集的第一个样本。
        
    Returns:
        所有任务的测试结果字典 {task_type: {model_name: results}}
    """
    all_results = {}
    tasks_to_run = []
    
    # 如果选择测试所有数据集，则为每个任务类型获取所有可用数据集
    if test_all_datasets:
        for task_type in task_types:
            # 获取该任务类型下的所有数据集
            available_datasets = get_available_datasets(task_type)
            if not available_datasets:
                print(f"警告: 任务类型 {task_type} 没有找到可用的数据集，将跳过")
                continue
                
            print(f"任务类型 {task_type} 找到 {len(available_datasets)} 个数据集")
            
            # 获取API密钥
            if api_keys and task_types.index(task_type) < len(api_keys) and api_keys[task_types.index(task_type)]:
                api_key = api_keys[task_types.index(task_type)]
            elif task_type in DEFAULT_API_KEYS:
                api_key = DEFAULT_API_KEYS[task_type]
            else:
                print(f"警告: 任务 {task_type} 没有找到对应的API密钥，将使用环境变量中的API_KEY")
                api_key = API_KEY  # 使用全局配置中的API密钥
                
            # 为每个数据集创建一个任务
            for dataset_name in available_datasets:
                tasks_to_run.append({
                    'task_type': task_type,
                    'dataset_name': dataset_name,
                    'api_key': api_key
                })
    # 使用指定的数据集列表
    else:
        if not dataset_names or len(task_types) != len(dataset_names):
            print("错误: 需要为每个任务类型提供对应的数据集名称")
            return all_results
            
        for i, task_type in enumerate(task_types):
            dataset_name = dataset_names[i]
            
            # 获取API密钥
            if api_keys and i < len(api_keys) and api_keys[i]:
                api_key = api_keys[i]
            elif task_type in DEFAULT_API_KEYS:
                api_key = DEFAULT_API_KEYS[task_type]
            else:
                print(f"警告: 任务 {task_type} 没有找到对应的API密钥，将使用环境变量中的API_KEY")
                api_key = API_KEY
                
            tasks_to_run.append({
                'task_type': task_type,
                'dataset_name': dataset_name,
                'api_key': api_key
            })
    
    print(f"开始多任务并行测试，共 {len(models)} 个模型，{len(tasks_to_run)} 个任务")
    for i, task in enumerate(tasks_to_run):
        print(f"  任务 {i+1}: {task['task_type']}, 数据集: {task['dataset_name']}")
    
    start_time = time.time()
    
    # 确保输出目录存在
    if save_results:
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if latest_dir and not os.path.exists(latest_dir):
            os.makedirs(latest_dir, exist_ok=True)
        if archive_dir and not os.path.exists(archive_dir):
            os.makedirs(archive_dir, exist_ok=True)
    
    # 创建并执行每个任务的进程
    processes = []
    for task in tasks_to_run:
        task_type = task['task_type']
        dataset_name = task['dataset_name']
        api_key = task['api_key']
        
        # 设置特定任务的环境变量 API_KEY
        env = os.environ.copy()
        env["API_KEY"] = api_key
        
        # 准备命令行参数 - 不再传递API密钥
        cmd_args = [
            sys.executable, __file__,  # 当前脚本
            "--multimodal",
            "--task_type", task_type,
            "--dataset_name", dataset_name,
            "--parallel_async",
            "--max_tokens", str(max_tokens),
            "--temperature", str(temperature),
            "--output_dir", f"{output_dir}/{task_type}/{dataset_name}",
            "--max_concurrent", str(max_concurrent_requests)
        ]
        
        # 确保任务输出目录存在
        task_output_dir = f"{output_dir}/{task_type}/{dataset_name}"
        if not os.path.exists(task_output_dir):
            os.makedirs(task_output_dir, exist_ok=True)
        
        # 添加样本限制参数
        if sample_limit is not None:
            cmd_args.extend(["--limit", str(sample_limit)])
        
        # 添加模型参数
        for model in models:
            cmd_args.extend(["--model", model])
        
        # 添加可选参数
        if latest_dir:
            task_latest_dir = f"{latest_dir}/{task_type}/{dataset_name}"
            cmd_args.extend(["--latest_dir", task_latest_dir])
            # 确保目录存在
            if not os.path.exists(task_latest_dir):
                os.makedirs(task_latest_dir, exist_ok=True)
        
        if archive_dir:
            task_archive_dir = f"{archive_dir}/{task_type}/{dataset_name}"
            cmd_args.extend(["--archive_dir", task_archive_dir])
            # 确保目录存在
            if not os.path.exists(task_archive_dir):
                os.makedirs(task_archive_dir, exist_ok=True)
        
        # 启动子进程
        print(f"启动任务 {task_type}/{dataset_name} 的子进程，使用API Key: {api_key[:5]}...{api_key[-5:]}")
        if sample_limit:
            print(f"  将只测试数据集的前 {sample_limit} 个样本")
            
        process = multiprocessing.Process(
            target=lambda: os.execve(sys.executable, cmd_args, env)
        )
        process.start()
        processes.append((process, f"{task_type}/{dataset_name}"))
    
    # 等待所有进程完成
    for process, task_id in processes:
        process.join()
        print(f"任务 {task_id} 已完成")
    
    total_time = time.time() - start_time
    print(f"所有任务并行测试完成，总用时: {total_time:.2f} 秒")
    
    # 收集结果
    for task in tasks_to_run:
        task_type = task['task_type']
        dataset_name = task['dataset_name']
        
        if task_type not in all_results:
            all_results[task_type] = {}
            
        if dataset_name not in all_results[task_type]:
            all_results[task_type][dataset_name] = {}
            
        task_output_dir = f"{output_dir}/{task_type}/{dataset_name}"
        if os.path.exists(task_output_dir):
            # 尝试读取每个模型的结果
            for model in models:
                model_dir = os.path.join(task_output_dir, model)
                if os.path.exists(model_dir):
                    all_results[task_type][dataset_name][model] = "完成"
                else:
                    all_results[task_type][dataset_name][model] = "未找到结果"
    
    return all_results

def load_config(config_path):
    """
    加载模型配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置对象
    """
    try:
        # 修正配置文件路径
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(__file__), config_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置文件失败: {str(e)}")
        # 如果加载失败，返回一个只包含gemini-2.5-pro-preview-03-25的配置
        return {
            "llmModels": [
                {
                    "name": "gemini-2.5-pro-preview-03-25",
                    "api_key": "sk-oNXIBwXUsoMVeCEU9tW66vkJsYatcqk9xExnrvfXMPUxZoel"
                }
            ]
        }

def get_model_api_name(model_name, model_config=None):
    """
    获取模型的API名称和API KEY
    
    Args:
        model_name: 模型名称
        model_config: 模型配置对象
        
    Returns:
        (api_model_name, api_key) 元组
    """
    # 如果提供了模型配置，从配置中查找
    if model_config and "llmModels" in model_config:
        for model_info in model_config["llmModels"]:
            if model_info.get("name") == model_name:
                # 在配置中找到模型名称和API KEY
                api_model_name = model_info.get("model", model_name)
                api_key = model_info.get("api_key")
                if api_key:
                    print(f"从配置中找到模型 {model_name} 的API KEY")
                    return api_model_name, api_key
    
    # 如果在默认配置中找到
    if model_name in MODELS:
        print(f"从默认配置中找到模型 {model_name} 的API KEY")
        return MODELS[model_name]["name"], MODELS[model_name]["api_key"]
    
    # 如果都没找到，返回原始名称和None
    print(f"未找到模型 {model_name} 的API KEY")
    return model_name, None

def call_model(model_name, message, system_prompt=None, max_tokens=800, temperature=0.7, model_config=None):
    """
    调用模型API获取响应
    
    Args:
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
    
    # 打印发送给API的消息
    print("\n===== 发送给API的消息 =====")
    print(json.dumps(formatted_messages, ensure_ascii=False, indent=2))
    print("="*30)
    
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
    model_config = load_config(args.config)
    
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