from .image_base_0302 import ImageBaseDataset
import pandas as pd
from .utils import build_judge, DEBUG_MESSAGE, extract_json_data
from ..smp import *
import ast
import re
import random
import json
from rich import print_json
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import datetime
from json_repair import repair_json



def cla_eval(data, eval_file):
    failed_items = 0
    y_true = []
    y_pred = []

    for i, row in data.iterrows():
        raw_prediction = row['prediction']
        pred = extract_json_data(i, raw_prediction)
        
        if pred == 'failed':
            failed_items += 1
            continue
        else:
            prediction_tmp_dict = pred
        
        prediction_dict = {
            'class': next(
                (prediction_tmp_dict[key] for key in ['class', 'answer', 'diagnosis'] if key in prediction_tmp_dict),
                None  # 所有键都不存在时的默认值
            )
        }

        data.at[i, 'predicted_class'] = prediction_dict['class']
        y_true.append(row['cla_ans'])
        y_pred.append(prediction_dict['class'])

    total_samples = len(data) - failed_items
    parser_rate = total_samples / len(data)
    accuracy = sum(np.array(y_true) == np.array(y_pred)) / total_samples
    precision = precision_score(y_true, y_pred, average='macro')  # 多分类用macro平均
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    res_dict = {
        'parser_rate': parser_rate,
        'acc': accuracy,
        'precision': precision,
        'recall': recall,
        'f1-score': f1}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    updated_eval_file = eval_file.replace('.xlsx', f'_eval_{timestamp}.xlsx')
    data.to_excel(updated_eval_file, index=False)  

    return res_dict


class ClaDataset(ImageBaseDataset):
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

        json_prompt_path = "cla_prompt.json"
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
        import sys
        data = load(eval_file)
        data = data.sort_values(by='index')
        if task_type == 'cla':
            acc_dict = cla_eval(data, eval_file)
            return acc_dict
       