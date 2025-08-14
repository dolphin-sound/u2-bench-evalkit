from .image_base_0302 import ImageBaseDataset
import pandas as pd
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
import ast
import re
import random
import json
import numpy as np
import pingouin as pg
from scipy import stats

def meaeval(data: pd.DataFrame, metric, **judge_kwargs) -> dict:
    """
    多任务评估核心函数
    Args:
        data: 必须包含 gt_ans (真实值) 和 prediction (预测值) 列
        task_type: 
            'cla' - 分类任务 (需prediction为类别)
            'reg' - 回归任务 (需prediction为连续值)
        judge_kwargs: 可传递 tolerance (允许误差阈值)
    Returns:
        包含 MAE, RMSE, Std, %_within_tolerance, ICC 的字典
    """
    
    # 预处理：过滤无效预测
    valid_data = data[
        (data['prediction'].notnull()) & 
        (~data['prediction'].isin(['Failed to obtain answer via API.', '']))
    ].copy()
    
    task_type = 'cla' if valid_data['prediction'][metric].astype(str) else 'reg'
    # 分类任务特殊处理
    if task_type == 'cla':
        y_true = valid_data['measurement_ans'][metric]
        y_pred = valid_data['prediction'][metric]

        return {
            'accuracy': np.mean(y_true == y_pred),
            'f1': stats.f1_score(y_true, y_pred, average='macro')
        }
    
    # 回归任务指标计算
    else:  
        # 类型转换与校验
        try:
            y_true = pd.to_numeric(valid_data['measurement_ans'][metric]).values
            y_pred = pd.to_numeric(valid_data['prediction'][metric]).values
        except ValueError:
            raise TypeError("回归任务要求 gt_ans 和 prediction 为数值类型")
        
        # 基础指标计算
        errors = y_true - y_pred
        tolerance = judge_kwargs.get('tolerance', 0.5)  # 默认阈值0.5
        
        metrics = {
            'MAE': np.mean(np.abs(errors)),
            'RMSE': np.sqrt(np.mean(errors**2)),
            'Std': np.std(errors),
            '%_within_tolerance': np.mean(np.abs(errors) <= tolerance) * 100
        }
        
        # ICC计算（需构造评分者矩阵）
        df_long = pd.DataFrame({
            'targets': np.repeat(valid_data.index, 2),
            'rater': ['true'] * len(y_true) + ['pred'] * len(y_pred),
            'score': np.concatenate([y_true, y_pred])
        })
        
        try:
            icc = pg.intraclass_corr(
                data=df_long, 
                targets='targets', 
                raters='rater', 
                ratings='score'
            ).set_index('Type')
            metrics['ICC(3,1)'] = icc.loc['ICC3k', 'ICC']  # 固定评分者绝对一致性
        except:
            metrics['ICC(3,1)'] = None  # 数据不足时返回空值
            
        return metrics

class MeasureDataset(ImageBaseDataset):
    TYPE = 'PENDING'
    MODALITY = 'IMAGE'

    TSV_PATH = {
        "put your TSV here"
    }

    def build_prompt(self, line, task_type):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['img_path'])
        else:
            tgt_path = self.dump_image(line)

        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        
        anatomy = line['anatomy_location']
            
        json_prompt_path = './mea_prompt.json'
        with open(json_prompt_path, 'r', encoding='utf-8') as f: 
            prompt_dict = json.load(f)
        prompt = prompt_dict[self.dataset_name]

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, task_type = 'cla', **judge_kwargs):
        data = load(eval_file)
        data = data.sort_values(by='index')
        measurement_dict = {}
        for metric in self.measurement_list:
            metric_dict_values = meaeval(data)
            measurement_dict[metric] = metric_dict_values
   
        return measurement_dict