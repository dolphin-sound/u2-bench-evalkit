import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime

TSV_PATH = {
        '03': '/media/ps/data-ssd/json_processing/ale_tsv_output/classification_tsv_output/3_FETAL_Planes_US.tsv',
        '37': '/media/ps/data-ssd/json_processing/ale_tsv_output/37.tsv',
        '50': '/media/ps/data-ssd/json_processing/ale_tsv_output/classification_tsv_output/50.ACOUSLIC_AI_Key_Frame_Classification_is_optimal_or_suboptimal.tsv',
        '53': '/media/ps/data-ssd/json_processing/ale_tsv_output/53.tsv',
        '69': '/media/ps/data-ssd/json_processing/ale_tsv_output/classification_tsv_output/69.tsv',
        '10': '/media/ps/data-ssd/json_processing/ale_tsv_output/classification_tsv_output/10_FetusOrientation.tsv',
        '18': '/media/ps/data-ssd/json_processing/ale_tsv_output/classification_tsv_output/18_1.Ultrasound_Heart_Segmentation_Dataset_view.tsv',
        '21': '/media/ps/data-ssd/json_processing/ale_tsv_output/21.Breast_Ultrasound_Segmentation_Dataset.tsv',
        '23': '/media/ps/data-ssd/json_processing/ale_tsv_output/23.tsv',
        '25': '/media/ps/data-ssd/json_processing/ale_tsv_output/25.Dermatologic_Ultrasound_Classification_Dataset.tsv',
        '28': '/media/ps/data-ssd/json_processing/ale_tsv_output/classification_tsv_output/28_1.Knee_Grading_Ultrasound_Classification_Dataset_Kellgren-Lawrence (KL) Grade.tsv',
        '32': '/media/ps/data-ssd/json_processing/ale_tsv_output/32_image.tsv',
        '40': '/media/ps/data-ssd/json_processing/ale_tsv_output/classification_tsv_output/40_birads.tsv',
        '42': '/media/ps/data-ssd/json_processing/ale_tsv_output/42.tsv',
        '44': '/media/ps/data-ssd/json_processing/ale_tsv_output/44.COVID-BLUES-frames.tsv',
        '57': '/media/ps/data-ssd/json_processing/ale_tsv_output/57_1.Liver_Ultrasound_Segmentation_Dataset.tsv',
        '66': '/media/ps/data-ssd/json_processing/ale_tsv_output/66.tsv',
        '70': '/media/ps/data-ssd/json_processing/ale_tsv_output/70.tsv',
        '75': '/media/ps/data-ssd/json_processing/ale_tsv_output/75.PCOS_Ultrasound_Classification_Dataset.tsv',
        '74is_normal': '/media/ps/data-ssd/json_processing/ale_tsv_output/classification_tsv_output/74_is_normal.tsv',
        '74is_visible': '/media/ps/data-ssd/json_processing/ale_tsv_output/classification_tsv_output/74_is_visible.tsv',
        'anatomy': '/media/ps/data-ssd/json_processing/ale_tsv_output/classification_tsv_output/anatomy.tsv',
        '28_class': '/media/ps/data-ssd/json_processing/ale_tsv_output/28.Knee_Classification.tsv'
    }


def read_jsonl_with_tsv(jsonl_path, tsv_path):
    """读取JSONL文件并合并对应TSV文件的class值"""
    jsonl_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            jsonl_data.append(json.loads(line.strip()))
    df_jsonl = pd.DataFrame(jsonl_data)

    # 读取TSV
    df_tsv = pd.read_csv(tsv_path, sep='\t')

    # 按行号合并（假设行号严格对应）
    if 'id' not in df_jsonl.columns or 'id' not in df_tsv.columns:
        if 'classes' in df_tsv.columns:
            df_jsonl['cla_ans'] = df_tsv['classes'].values  
        else:
            df_jsonl['cla_ans'] = df_tsv['class'].values  
    else:
        # 按ID精确匹配
        df_merged = pd.merge(df_jsonl, df_tsv[['id', 'classes']], 
                           on='id', how='left', suffixes=('', '_tsv'))
        df_jsonl['cla_ans'] = df_merged['class']

    return df_jsonl

def model_eval(data):
    """更新后的评估函数"""
    failed_items = 0
    y_true = []
    y_pred = []
    for i, row in data.iterrows():
        if 'model' in data.keys() and data['model'][0] == 'LLaVA-1.5-13B-HF':
            pred = row['response'].replace(' ', '', 1)
        else:
            pred = row['response'] if row.get('response') else row['prediction']
        gt_key = next((key for key in ['classes', 'class', 'cla_ans'] if key in data.columns), None)
        if isinstance(row[gt_key], str):
            if '[' in row[gt_key]:
                row[gt_key] = row[gt_key].replace('[','').replace(']','')
            if isinstance(row[gt_key], int):
                y_true.append(row[gt_key])
                y_pred.append(int(pred))
            else:
                row[gt_key] = row[gt_key].replace("'",'')
                y_true.append(row[gt_key].lower())  
                y_pred.append(pred.lower())
        elif isinstance(row[gt_key], int):
            try:
                pred = int(pred)
            except:
                failed_items += 1
                continue
            y_true.append(row[gt_key]) 
            y_pred.append(pred)
        else:
            y_true.append(row[gt_key])  
            y_pred.append(pred)

    # 计算指标
    total_samples = len(data) - failed_items
    return {
        'parser_rate': total_samples / len(data),
        'acc': sum(np.array(y_true) == np.array(y_pred)) / total_samples,
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro')
    }

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

def find_model_xlsxs(base_dir, dataset_id):
    dataset_dir = os.path.join(base_dir, dataset_id)
    model_files = []
    
    for model_dir in os.listdir(dataset_dir):
        model_path = os.path.join(dataset_dir, model_dir)
        for task_dir in os.listdir(model_path):
            if task_dir.endswith('cla'):
                for file in os.listdir(os.path.join(model_path, task_dir)):
                    full_path = os.path.join(model_path, task_dir, file)
                    model_files.append(full_path)
    return model_files

def batch_evaluate_all_tasks():
    """批量处理所有任务的主函数"""
    base_dir = '/media/ps/data-ssd/benchmark/VLMEvalKit/outputs/dolphin-output/cla'
    output_file = f'cla_results_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
    
    all_results = []

    # 遍历所有预定义的任务
    for task_id in TSV_PATH.keys():
        # 自动发现该任务下的所有模型文件
        task_pairs = find_model_files(base_dir, task_id)
        
        if not task_pairs:
            print(f"警告：任务 {task_id} 未找到任何模型结果文件")
            continue
        
        # 执行评估
        task_results = evaluate_task(task_pairs, task_id)
        all_results.extend(task_results)
    
    # 写入统一结果文件
    write_combined_results(all_results, output_file)

def evaluate_task(file_pairs, task_id):
    """评估单个任务下的所有模型"""
    task_results = []
    
    for jsonl_path, tsv_path in file_pairs:
        try:
            model_name = jsonl_path.split('/')[-2]  # 根据路径结构调整索引
            data = read_jsonl_with_tsv(jsonl_path, tsv_path)

            metrics = model_eval(data)
            
            # 记录附加信息
            metrics.update({
                'task_id': task_id,
                'model': model_name,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            task_results.append(metrics)
            
        except Exception as e:
            print(f"处理文件失败：{jsonl_path}")
            print(f"错误信息：{str(e)}")
    
    return task_results

def write_combined_results(results, output_file):
    """写入整合后的结果"""
    columns = [
        'task_id', 'model', 'timestamp',
        'parser_rate', 'acc', 'precision', 'recall', 'f1'
    ]
    
    with open(output_file, 'w') as f:
        # 写入表头
        f.write("\t".join(columns) + "\n")
        
        # 写入数据行
        for res in results:
            line = "\t".join([
                res['task_id'],
                res['model'],
                res['timestamp'],
                f"{res['parser_rate']:.4f}",
                f"{res['acc']:.4f}",
                f"{res['precision']:.4f}",
                f"{res['recall']:.4f}",
                f"{res['f1']:.4f}"
            ]) + "\n"
            f.write(line)

def evaluate_task_with_xlsx(files):
    task_results = []
    
    for xlsx_path in files:
        try:
            model_name = xlsx_path.split('/')[-3]  # 根据路径结构调整索引
            data = pd.read_excel(xlsx_path)
            metrics = model_eval(data)

            task_id = xlsx_path.split('/')[-4]
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
    output_file = f'cla_results_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
    
    all_results = []

    for task_id in TSV_PATH.keys():
        task_pairs = find_model_xlsxs(base_dir, task_id)
        
        if not task_pairs:
            print(f"警告：任务 {task_id} 未找到任何模型结果文件")
            continue

        task_results = evaluate_task_with_xlsx(task_pairs)
        all_results.extend(task_results)

    write_combined_results(all_results, output_file)


if __name__ == "__main__":
    batch_evaluate_all_tasks_xlsx()