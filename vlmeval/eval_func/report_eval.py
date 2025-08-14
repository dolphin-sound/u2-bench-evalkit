import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pycocoevalcap.bleu.bleu import Bleu 
from pycocoevalcap.rouge.rouge import Rouge 
from transformers import AutoTokenizer, AutoModel
import torch

# 配置参数
HAVE_LOCAL_BERTSCORE_MODEL = False
LOCAL_MODEL_DIR = "/path/to/models"
BERTSCORE_MODEL_NAME = "bert-base-multilingual-cased"
TSV_PATH = {
    '10': '/media/ps/data-ssd/json_processing/ale_tsv_output/10.tsv',
    '11': '/media/ps/data-ssd/json_processing/ale_tsv_output/11.Thyroid_US_Images.tsv',
    '39': '/media/ps/data-ssd/json_processing/ale_tsv_output/39_translated.tsv',
    # '44': '/media/ps/data-ssd/json_processing/ale_tsv_output/44.COVID-BLUES-frames.tsv',
}

# 单例模型缓存
class BertModelCache:
    _instance = None

    def __init__(self):
        if self._instance is not None:
            raise Exception("单例模式，请使用 instance() 方法")
            
        self.tokenizer, self.model, self.device = self.load_bert_model()
        BertModelCache._instance = self

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = BertModelCache()
        return cls._instance

    def load_bert_model(self):
        """优化后的模型加载方法"""
        try:
            if HAVE_LOCAL_BERTSCORE_MODEL:
                model_path = f"{LOCAL_MODEL_DIR}/{BERTSCORE_MODEL_NAME}"
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModel.from_pretrained(model_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
                model = AutoModel.from_pretrained("bert-base-multilingual-cased")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            return tokenizer, model, device
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return None, None, "cpu"

    @classmethod
    def clear_cache(cls):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def batch_bertscore(refs_dict, hyps_dict):
    """批量计算BERTScore"""
    cache = BertModelCache.instance()
    scores = {}
    
    try:
        import bert_score
        calc_params = {
            "model_type": "bert-base-multilingual-cased",
            "lang": "en",
            "device": cache.device,
            "batch_size": 8  
        }

        # 构建输入列表
        all_hyp, all_ref = [], []
        for case_id, hyp in hyps_dict.items():
            ref = refs_dict.get(case_id, [])
            if ref:
                all_hyp.extend(hyp)
                all_ref.extend(ref)

        # 批量计算
        if len(all_hyp) > 0:
            _, _, F1 = bert_score.score(
                all_hyp, all_ref,
                **calc_params
            )
            
            # 按案例聚合分数
            idx = 0
            for case_id, hyp in hyps_dict.items():
                refs = refs_dict.get(case_id, [])
                if refs:
                    scores[case_id] = F1[idx:idx+len(refs)].mean().item()
                    idx += len(refs)
    except Exception as e:
        print(f"BERTScore批量计算失败: {str(e)}")
    finally:
        BertModelCache.clear_cache()
    
    return scores

class MedicalReportScorer:
    def __init__(self, refs, hyps):
        self.refs = refs
        self.hyps = hyps
        self.scorers = [
            (Bleu(4), 'Bleu'),
            (Rouge(), 'Rouge')
        ]

    def evaluate(self):
        metrics = {}
        for scorer, name in self.scorers:
            try:
                score, _ = scorer.compute_score(self.refs, self.hyps)
                if name == 'Bleu':
                    for i, v in enumerate(score):
                        metrics[f'Bleu-{i+1}'] = v * 100
                else:
                    metrics[name] = score * 100
            except Exception as e:
                print(f"{name}计算失败: {str(e)}")

        # BERTScore
        try:
            bert_scores = batch_bertscore(self.refs, self.hyps)
            metrics['BERTScore'] = np.mean(list(bert_scores.values())) * 100
        except Exception as e:
            print(f"BERTScore计算失败: {str(e)}")
            metrics['BERTScore'] = 0.0

        return metrics
    
def find_model_files(base_dir, task_id):
    """自动发现指定任务下的所有模型结果文件"""
    task_dir = os.path.join(base_dir, task_id)
    model_files = []
    
    for model_dir in os.listdir(task_dir):
        model_path = os.path.join(task_dir, model_dir)
        if os.path.isdir(model_path):
            for file in os.listdir(model_path):
                if file.startswith('multimodal_test_results') and file.endswith('.jsonl'):
                    full_path = os.path.join(model_path, file)
                    model_files.append((full_path, TSV_PATH[task_id]))
    return model_files

def read_jsonl_with_tsv(jsonl_path, tsv_path):
    """读取JSONL文件并合并TSV中的参考报告"""
    # 读取生成报告
    with open(jsonl_path, 'r') as f:
        jsonl_data = [json.loads(line) for line in f]
    df_jsonl = pd.DataFrame(jsonl_data)

    df_tsv = pd.read_csv(tsv_path, sep='\t')
    
    df_merged = df_jsonl.assign(cap_ans=df_tsv['caption'])
    return df_merged

def report_eval(data):
    """医疗报告评估函数"""
    # try:
    gt_key = 'caption' if 'caption' in data.keys() else 'cap_ans'
    id_key = 'id' if 'id' in data.keys() else 'index'
    pred_key = 'response' if 'response' in data.keys() else 'prediction'
    refs = {row[id_key]: [row[gt_key]] for _, row in data.iterrows()}
    hyps = {row[id_key]: [row[pred_key]] for _, row in data.iterrows()}

    scorer = MedicalReportScorer(refs, hyps)
    metrics = scorer.evaluate()
    
    return metrics
    # except Exception as e:
    #     return {'error': f"Evaluation failed: {str(e)}"}

def batch_evaluate_all_tasks():
    base_dir = '/media/ps/data-ssd/benchmark/VLMEvalKit/outputs/dolphin-output-0512/dolphin-output/report'
    output_file = f'report_results_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
    
    all_results = []
    
    for task_id in TSV_PATH:
        task_pairs = find_model_files(base_dir, task_id)
        if not task_pairs:
            print(f"Task {task_id} skipped: no files found")
            continue
        
        for jsonl_path, tsv_path in task_pairs:
            # try:
            # 读取数据
            data = read_jsonl_with_tsv(jsonl_path, tsv_path)
            if data.empty:
                print(f"Skipped {jsonl_path}: empty data")
                continue
            
            # 从路径解析模型名称
            model_name = jsonl_path.split('/')[-2]
            
            # 执行评估
            metrics = report_eval(data)
            if 'error' in metrics:
                print(f"Skipped {model_name}@{task_id}: {metrics['error']}")
                continue
            
            # 记录结果
            all_results.append({
                'task_id': task_id,
                'model': model_name,
                **metrics
            })
            print(f"Success to process {jsonl_path}")
            # except Exception as e:
            #     print(f"Failed to process {jsonl_path}: {str(e)}")
    
    # 保存结果
    save_results(all_results, output_file)
    print(f"Evaluation completed. Results saved to {output_file}")

def save_results(results, output_file):
    """更新后的结果保存函数"""
    columns = [
        'task_id', 'model',
        'Bleu-4',
        'Rouge', 'BERTScore'
    ]
    
    with open(output_file, 'w') as f:
        # 表头
        f.write('\t'.join(columns) + '\n')
        
        # 数据行
        for res in results:
            line = '\t'.join([
                res['task_id'],
                res['model'],
                f"{res.get('Bleu-4', 0):.2f}",
                f"{res.get('Rouge', 0):.2f}",
                f"{res.get('BERTScore', 0):.2f}"
            ]) + '\n'
            f.write(line)


def find_model_xlsxs(base_dir, dataset_id):
    dataset_dir = os.path.join(base_dir, dataset_id)
    model_files = []
    
    for model_dir in os.listdir(dataset_dir):
        model_path = os.path.join(dataset_dir, model_dir)
        for task_dir in os.listdir(model_path):
            if task_dir.endswith('report'):
                for file in os.listdir(os.path.join(model_path, task_dir)):
                    full_path = os.path.join(model_path, task_dir, file)
                    model_files.append(full_path)
    return model_files

def evaluate_task_with_xlsx(files):
    task_results = []
    
    for xlsx_path in files:
        try:
            model_name = xlsx_path.split('/')[-3]  # 根据路径结构调整索引
            task_id = xlsx_path.split('/')[-4]
            
            if model_name == 'MiniGPT-med':
                data = pd.read_excel(xlsx_path)
                metrics = report_eval(data)

                metrics.update({
                    'task_id': task_id,
                    'model': model_name,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                task_results.append(metrics)
            
        except Exception as e:
            print(f"处理文件失败：{xlsx_path}")
            print(f"错误信息：{str(e)}")
    
    return task_results

def batch_evaluate_all_tasks_xlsx():
    base_dir = '/media/ps/data-ssd/benchmark/VLMEvalKit/outputs/'
    output_file = f'report_results_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
    
    all_results = []

    for task_id in TSV_PATH.keys():
        task_pairs = find_model_xlsxs(base_dir, task_id)
        
        if not task_pairs:
            print(f"警告：任务 {task_id} 未找到任何模型结果文件")
            continue

        task_results = evaluate_task_with_xlsx(task_pairs)
        all_results.extend(task_results)

    save_results(all_results, output_file)

if __name__ == "__main__":
    batch_evaluate_all_tasks_xlsx()