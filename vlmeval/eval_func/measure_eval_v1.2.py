import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
import pingouin as pg

# 预定义的TSV路径映射, 31单独测评
TSV_PATH = {
        '18': '/media/ps/data-ssd/json_processing/ale_tsv_output/18_1.Ultrasound_Heart_Segmentation_Dataset.tsv',
        '27': '/media/ps/data-ssd/json_processing/ale_tsv_output/27.tsv',
        # '31': '/media/ps/data-ssd/json_processing/ale_tsv_output/31.tsv',
        '50': '/media/ps/data-ssd/json_processing/ale_tsv_output/50.tsv',
        '57': '/media/ps/data-ssd/json_processing/ale_tsv_output/57_1.Liver_Ultrasound_Segmentation_Dataset.tsv'
    }

MIN_MAX_DICT = {
    '18': {'min': 10.0, 'max': 75.0},
    '27': {'min': 0.3, 'max': 1.5},
    '50': {'min': 100.0, 'max': 400.0},
    '57': {'min': 0, 'max': 85},
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


def read_jsonl_with_tsv(jsonl_path, tsv_path, task_id):
    """读取JSONL文件并合并TSV中的ans"""
    def extract_mea(measurements_str):
        try:
            fixed_str = measurements_str.replace("'", '"')
            data = json.loads(fixed_str)
            if isinstance(data, dict):
                if task_id == '18':
                    return data['EF']
                elif task_id == '27':
                    return list(data['IMT'])[0]
                elif task_id == '31':
                    return data
                elif task_id == '50':
                    return data['abdominal_circumference']
                else:
                    return data['fat value']
            elif isinstance(data, list):
                return list(data[0].values())[0]
        except:
            return 'nan' 

    with open(jsonl_path, 'r') as f:
        jsonl_data = [json.loads(line) for line in f]
    df_jsonl = pd.DataFrame(jsonl_data)

    df_tsv = pd.read_csv(tsv_path, sep='\t')

    if 'id' in df_jsonl.columns and 'id' in df_tsv.columns:
        df_merged = pd.merge(df_jsonl, df_tsv[['id', 'measurement_ans']], on='id', how='left')
    else:
        df_merged = df_jsonl.assign(measurement_ans=df_tsv['measurement'].apply(extract_mea))
    return df_merged

def min_max_scale(y_arr, task_id):
    min_val = MIN_MAX_DICT[task_id]['min']
    max_val = MIN_MAX_DICT[task_id]['max']

    if max_val == min_val:
        return np.zeros_like(y_arr, dtype=float)
    else:
        return (y_arr - min_val) / (max_val - min_val)
    
def meaeval(data: pd.DataFrame, task_id: str, **judge_kwargs) -> dict:  
    def extract_mea(measurements_str):
        try:
            fixed_str = measurements_str.replace("'", '"')
            mdata = json.loads(fixed_str)
            if isinstance(mdata, dict):
                if task_id == '18':
                    return mdata['EF']
                elif task_id == '27':
                    return list(mdata['IMT'])[0]
                elif task_id == '31':
                    return mdata
                elif task_id == '50':
                    return mdata['abdominal_circumference']
                else:
                    return mdata['fat value']
            elif isinstance(mdata, list):
                return list(mdata[0].values())[0]
        except:
            return 'nan' 
    
    y_true = []
    y_pred = []
    failed_items = 0
    for _, row in data.iterrows():
        gt_key = 'measurement_ans' if 'measurement_ans' in data.keys() else 'measurement' 
        if isinstance(row[gt_key], list):
            gt = row[gt_key][0]
        else:
            gt = row[gt_key]
        gt = extract_mea(gt)
        if gt == 'nan' or gt is None:
            continue

        if 'model' in data.keys() and data['model'][0] == 'LLaVA-1.5-13B-HF':
            pred = row['response'].replace(' ', '', 1)
        else:
            pred_key = 'response' if 'response' in data.keys() else 'prediction'
            pred = row[pred_key]

        try :
            pred = float(pred)
        except:
            failed_items += 1
            continue
        
        y_true.append(float(gt))
        y_pred.append(pred)
        print(f'task: {task_id}, gt: {gt}, pred: {pred}')
    # 添加正则化
    y_true = np.array(y_true)
    y_true = min_max_scale(y_true, task_id)
    y_pred = np.array(y_pred)
    y_pred = min_max_scale(y_pred, task_id)
    errors = y_true - y_pred
    tolerance = judge_kwargs.get('tolerance', 0.1)
    
    metrics = {
        'MAE': np.mean(np.abs(errors)),
        'RMSE': np.sqrt(np.mean(errors**2)),
        'Std': np.std(errors),
        '%_within_tolerance': np.mean(np.abs(errors) <= tolerance) * 100,
        'failed_items': failed_items
    }
    
    # # ICC计算（需构造评分者矩阵）
    # icc_data = pd.DataFrame({
    #         'target': np.repeat(np.arange(len(y_true)), 2),
    #         'rater': np.tile(['true', 'pred'], len(y_true)),
    #         'score': np.column_stack([y_true, y_pred]).flatten()
    #     })
    
    # try:
    #     icc_result = pg.intraclass_corr(
    #         data=icc_data,
    #         targets='target',
    #         raters='rater',
    #         ratings='score'
    #     ).set_index('Type')
        
    #     metrics['ICC(3,1)'] = icc_result.loc['ICC3k', 'ICC']
    # except:
    #     metrics['ICC(3,1)'] = None  # 数据不足时返回空值
            
    return metrics

def batch_evaluate_all_tasks():
    """批量处理所有任务的主函数"""
    base_dir = '/media/ps/data-ssd/benchmark/VLMEvalKit/outputs/dolphin-output-0512/dolphin-output/measurement'
    output_file = f'mea_results_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
    
    all_results = []
    
    for task_id in TSV_PATH:
        task_pairs = find_model_files(base_dir, task_id)
        if not task_pairs:
            print(f"Task {task_id} skipped: no files found")
            continue
        
        for jsonl_path, tsv_path in task_pairs:
            # try:
            # 读取数据
                data = read_jsonl_with_tsv(jsonl_path, tsv_path, task_id)

                # 从路径解析模型名称
                model_name = jsonl_path.split('/')[-2]  # 根据实际路径结构调整
                
                # 执行评估
                if task_id == '31':
                   continue
                else:
                    metrics = meaeval(data, task_id)
                    if 'error' in metrics:
                        print(f"Skipped {model_name}@{task_id}: {metrics['error']}")
                        continue

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
    """保存评估结果"""
    columns = ['task_id', 'model', 'RMSE', 'MAE', 'Std', '%_within_tolerance', 'failed_items']
    
    with open(output_file, 'w') as f:
        # 表头
        f.write('\t'.join(columns) + '\n')
        
        # 数据行
        for res in results:
            line = '\t'.join([
                res['task_id'],
                res['model'],
                f"{res['RMSE']:.4f}",
                f"{res['MAE']:.4f}",
                f"{res['Std']:.4f}",
                f"{res['%_within_tolerance']:.4f}",
                f"{res['failed_items']}"
            ]) + '\n'
            f.write(line)


def find_model_xlsxs(base_dir, dataset_id):
    dataset_dir = os.path.join(base_dir, dataset_id)
    model_files = []
    
    for model_dir in os.listdir(dataset_dir):
        model_path = os.path.join(dataset_dir, model_dir)
        for task_dir in os.listdir(model_path):
            if task_dir.endswith('measurement'):
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
            
            data = pd.read_excel(xlsx_path)
            metrics = meaeval(data, task_id)

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
    output_file = f'measure_results_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
    
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