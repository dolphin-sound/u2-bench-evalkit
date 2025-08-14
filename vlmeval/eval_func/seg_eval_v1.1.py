import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime

# 预定义的TSV路径映射
TSV_PATH = {
        '04': '/media/ps/data-ssd/json_processing/ale_tsv_output/04.tsv',
        '17': '/media/ps/data-ssd/json_processing/ale_tsv_output/17_1.tsv',
        '23': '/media/ps/data-ssd/json_processing/ale_tsv_output/23.tsv',
        '32': '/media/ps/data-ssd/json_processing/ale_tsv_output/32_image.tsv',
        '64': '/media/ps/data-ssd/json_processing/ale_tsv_output/64.BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.tsv',

        '09': '/media/ps/data-ssd/json_processing/ale_tsv_output/single_channel_seg/09.tsv',
        '13': '/media/ps/data-ssd/json_processing/ale_tsv_output/single_channel_seg/13.tsv',
        '16': '/media/ps/data-ssd/json_processing/ale_tsv_output/single_channel_seg/16.tsv',
        '18': '/media/ps/data-ssd/json_processing/ale_tsv_output/single_channel_seg/18_1.Ultrasound_Heart_Segmentation_Dataset.tsv',
        '31': '/media/ps/data-ssd/json_processing/ale_tsv_output/31.tsv',
        '37': '/media/ps/data-ssd/json_processing/ale_tsv_output/single_channel_seg/37.tsv',
        '38': '/media/ps/data-ssd/json_processing/ale_tsv_output/38.tsv',
        '47': '/media/ps/data-ssd/json_processing/ale_tsv_output/47.tsv',
        '49': '/media/ps/data-ssd/json_processing/ale_tsv_output/49.tsv',
        '50': '/media/ps/data-ssd/json_processing/ale_tsv_output/50.tsv',
        '52': '/media/ps/data-ssd/json_processing/ale_tsv_output/single_channel_seg/52.tsv',
        '53': '/media/ps/data-ssd/json_processing/ale_tsv_output/single_channel_seg/53.tsv',
        '67': '/media/ps/data-ssd/json_processing/ale_tsv_output/single_channel_seg/67.tsv',

        '48': '/media/ps/data-ssd/json_processing/ale_tsv_output/single_keypoint/48.tsv'
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


def read_jsonl_with_tsv(jsonl_path, tsv_path):
    """读取JSONL文件并合并TSV中的seg_ans"""
    def extract_coordinates(keypoints_str):
        try:
            fixed_str = keypoints_str.replace("'", '"')
            data = json.loads(fixed_str)
            if isinstance(data, dict):
                return list(data.values())[0] # 这种情况下，默认只有一对坐标需要提取
            else:
                return list(data[0].values())[0] # 只取第一个
        except:
            return 'nan' 
    with open(jsonl_path, 'r') as f:
        jsonl_data = [json.loads(line) for line in f]
    df_jsonl = pd.DataFrame(jsonl_data)

    df_tsv = pd.read_csv(tsv_path, sep='\t')
    
    # 按ID匹配或行号匹配
    if 'id' in df_jsonl.columns and 'id' in df_tsv.columns:
        df_merged = pd.merge(df_jsonl, df_tsv[['id', 'seg_ans']], on='id', how='left')
    else:
        if 'keypoints' in df_tsv.columns:
            df_merged = df_jsonl.assign(seg_ans=df_tsv['keypoints'].apply(extract_coordinates))
        elif 'gt_bbox' in df_tsv.columns:
            df_merged = df_jsonl.assign(seg_ans=df_tsv['gt_bbox'].apply(extract_coordinates))
    return df_merged


def get_bounding_box_location_v1(gt_bbox):
    """
    Determines the location category of a bounding box.
    """
    def extract_coordinates(keypoints_str):
        try:
            fixed_str = keypoints_str.replace("'", '"')
            data = json.loads(fixed_str)
            if isinstance(data, dict):
                return list(data.values())[0] # 这种情况下，默认只有一对坐标需要提取
            else:
                return list(data[0].values())[0] # 只取第一个
        except:
            return 'nan' 
    if isinstance(gt_bbox, str):
        gt_bbox = extract_coordinates(gt_bbox)
    import math
    if isinstance(gt_bbox, float) and math.isnan(gt_bbox): # For gt_bbox = float('nan')
        return 'not visible'
    
    # Check if gt_bbox is a sequence of expected length
    if not isinstance(gt_bbox, (list, tuple)) or len(gt_bbox) < 2: # Expecting at least center_x, center_y
        return 'not visible'

    try:
        center_x, center_y = float(gt_bbox[0]), float(gt_bbox[1])
    except (ValueError, TypeError): # Handles non-numeric values in gt_bbox
        return 'not visible'

    if math.isnan(center_x) or math.isnan(center_y):
        return 'not visible'

    LOW_THRESHOLD = 0.45
    HIGH_THRESHOLD = 0.55

    if center_y < LOW_THRESHOLD:
        y_pos = "upper"
    elif center_y < HIGH_THRESHOLD: 
        y_pos = "middle"
    else: 
        y_pos = "lower"

    if center_x < LOW_THRESHOLD:
        x_pos = "left"
    elif center_x < HIGH_THRESHOLD: 
        x_pos = "center"
    else: 
        x_pos = "right"

    if y_pos == "middle" and x_pos == "center":
        return "center"
    else:
        return f"{y_pos} {x_pos}"
    

def new_seg_eval(data):
    from sklearn.metrics import precision_score, recall_score, f1_score
    gt_list = []
    pred_list = []
    failed_items = 0
    for _, row in data.iterrows():
        gt_key = 'gt_bbox' if 'gt_bbox' in data.keys() else 'keypoints'
        gt_bbox = row[gt_key]
        gt = get_bounding_box_location_v1(gt_bbox)
        if 'model' in data.keys() and data['model'][0] == 'LLaVA-1.5-13B-HF':
            raw_pred = row['response'].replace(' ', '', 1)
            raw_pred = raw_pred.lower()
            # print(f'gt: {gt}; pred: {raw_pred}')
        else:
            raw_pred = row.get('response') if 'response' in data.keys() else row.get('prediction')
        if isinstance(raw_pred, str): raw_pred = raw_pred.lower()
        else: 
            failed_items += 1 
            continue
    
        # if pd.isna(raw_pred) or raw_pred == 'failed':
        #     failed_items += 1
        #     continue
        print(f'gt: {gt}; pred: {raw_pred}')
        gt_list.append(gt)
        pred_list.append(raw_pred)

    total_samples = len(data) - failed_items
    if total_samples == 0:
        return {'error': 'No valid samples'}
    return {
        # 'parser_rate': total_samples / len(data),
        'acc': sum(g == p for g, p in zip(gt_list, pred_list)) / total_samples,
        # 'precision': precision_score(gt_list, pred_list, average='macro'),
        # 'recall': recall_score(gt_list, pred_list, average='macro'),
        # 'f1': f1_score(gt_list, pred_list, average='macro')
    }

def batch_evaluate_all_tasks():
    """批量处理所有任务的主函数"""
    base_dir = '/media/ps/data-ssd/benchmark/VLMEvalKit/outputs/dolphin-output-0512/dolphin-output/seg'
    output_file = f'seg_results_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
    
    all_results = []
    
    for task_id in TSV_PATH:
        task_pairs = find_model_files(base_dir, task_id)
        if not task_pairs:
            print(f"Task {task_id} skipped: no files found")
            continue
        
        for jsonl_path, tsv_path in task_pairs:
            try:
                data = read_jsonl_with_tsv(jsonl_path, tsv_path)

                model_name = jsonl_path.split('/')[-2]  # 根据实际路径结构调整

                metrics = new_seg_eval(data)
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
            except Exception as e:
                print(f"Failed to process {jsonl_path}: {str(e)}")
    
    # 保存结果
    save_results(all_results, output_file)
    print(f"Evaluation completed. Results saved to {output_file}")

def save_results(results, output_file):
    """保存评估结果"""
    columns = ['task_id', 'model', 'acc']
    
    with open(output_file, 'w') as f:
        # 表头
        f.write('\t'.join(columns) + '\n')
        
        # 数据行
        for res in results:
            line = '\t'.join([
                res['task_id'],
                res['model'],
                # f"{res['parser_rate']:.4f}",
                f"{res['acc']:.4f}",
                # f"{res['precision']:.4f}",
                # f"{res['recall']:.4f}",
                # f"{res['f1']:.4f}"
            ]) + '\n'
            f.write(line)

def find_model_xlsxs(base_dir, dataset_id):
    dataset_dir = os.path.join(base_dir, dataset_id)
    model_files = []
    
    for model_dir in os.listdir(dataset_dir):
        model_path = os.path.join(dataset_dir, model_dir)
        for task_dir in os.listdir(model_path):
            if task_dir.endswith('seg'):
                for file in os.listdir(os.path.join(model_path, task_dir)):
                    full_path = os.path.join(model_path, task_dir, file)
                    model_files.append(full_path)
    return model_files

def evaluate_task_with_xlsx(files):
    task_results = []
    
    for xlsx_path in files:
        try:
            model_name = xlsx_path.split('/')[-3]  # 根据路径结构调整索引
            data = pd.read_excel(xlsx_path)
            metrics = new_seg_eval(data)

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
    output_file = f'seg_results_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
    
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