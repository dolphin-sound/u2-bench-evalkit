from .image_base_0302 import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE, extract_json_data
from ..smp import *
import ast
import re
import random
from rich import print_json
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torchvision.ops import nms
from sklearn.metrics import average_precision_score
import json

def _cxcywh_to_xyxy(boxes, img_w, img_h):
    """
    将 [x_center, y_center, w_rel, h_rel] 格式的框转换为 [x1, y1, x2, y2] 格式的绝对像素坐标。
    """
    boxes = np.array(boxes, dtype=float)
    cx, cy, w_rel, h_rel = boxes.T
    w = w_rel * img_w
    h = h_rel * img_h
    x1 = cx * img_w - w / 2
    y1 = cy * img_h - h / 2
    x2 = x1 + w
    y2 = y1 + h
    return np.stack([x1, y1, x2, y2], axis=1)

def _iou(boxA, boxB):
    """
    计算两个 [x1, y1, x2, y2] 格式的框之间的 IoU。
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interA = interW * interH
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    unionA = areaA + areaB - interA
    return interA / unionA if unionA > 0 else 0.0

def seg_eval(data, eval_file, iou_thr=0.5, img_size=(1.0, 1.0)):
    """
    评估分割模型的性能，计算 mean_dice、mAP 和 parser_rate。
    
    参数:
    - data: 包含每张图像元数据的 DataFrame，需包含 'prediction' 和 'gt_bboxes' 列。
    - eval_file: 结果保存的 Excel 文件路径，例如 'results.xlsx'。
    - iou_thr: IoU 阈值，用于确定匹配（默认值为 0.5）。
    - img_size: 图像的尺寸（宽度, 高度），用于将相对坐标转换为绝对坐标。
    
    返回:
    - 包含评估指标的字典。
    """
    failed_items = 0
    y_true = []
    y_scores = []
    dice_list = []
    parser_count = 0

    W, H = img_size

    for i, row in data.iterrows():
        raw_prediction = row['prediction']
        pred = extract_json_data(i, raw_prediction)

        if pred == 'failed':
            failed_items += 1
            continue
        else:
            bbox_list = []
            score_list = []
            if isinstance(pred, list):
                for item in len(pred):
                    bbox_list.append(item['bbox'])
                    score_list.append(item['score'])
            elif isinstance(pred, dict):
                bbox_list.append(item['bbox'])
                score_list.append(item['score'])

            # 将框转换为绝对坐标
            boxes = _cxcywh_to_xyxy([box for box in bbox_list], W, H)
            scores = np.array(score_list, dtype=float)

            # 应用非极大值抑制（NMS）
            keep_idx = nms(torch.tensor(boxes), torch.tensor(scores), iou_thr).numpy()
            kept_boxes = boxes[keep_idx]
            kept_scores = scores[keep_idx]

            # 获取当前图像的真实框
            gt_boxes = row['seg_ans'] if 'seg_ans' in row and isinstance(row['seg_ans'], list) else []

            # 对每个保留的预测框，计算与真实框的最大 IoU 和对应的 Dice 系数
            for box, sc in zip(kept_boxes, kept_scores):
                if gt_boxes:
                    ious = [_iou(box, gt) for gt in gt_boxes]
                    best_iou = max(ious)
                else:
                    best_iou = 0.0
                dice = 2 * best_iou / (1 + best_iou) if best_iou > 0 else 0.0

                dice_list.append(dice)
                y_true.append(int(best_iou >= iou_thr))
                y_scores.append(sc)

    # 计算评估指标
    mean_dice = float(np.mean(dice_list)) if dice_list else 0.0
    ap_score = float(average_precision_score(y_true, y_scores)) if any(y_true) else 0.0
    total_items = len(data)
    parsed_items = total_items - failed_items
    parser_rate = parsed_items / total_items if total_items > 0 else 0.0

    # 保存结果到 Excel 文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    updated_eval_file = eval_file.replace('.xlsx', f'_eval_{timestamp}.xlsx')

    # 创建包含评估指标的 DataFrame
    metrics_df = pd.DataFrame([{
        'mean_dice': mean_dice,
        'mAP': ap_score,
        'parser_rate': parser_rate,
        'failed_items': failed_items,
        'total_items': total_items
    }])

    # 将原始数据和评估指标合并
    result_df = pd.concat([data, metrics_df], axis=1)

    # 保存到 Excel 文件
    result_df.to_excel(updated_eval_file, index=False)
    
    # save into DataFrame and to Excel
    results = {
        'mean_dice': mean_dice,
        'mAP': ap_score,
        'parser_rate': parser_rate
    }
    df = df.assign(**results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = eval_file.replace('.xlsx', f'_eval_{timestamp}.xlsx')
    df.to_excel(out_file, index=False)
    
    return results


def new_seg_eval(data, eval_file):
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    gt_list = []
    pred_list = []
    failed_items = 0
    for i, row in data.iterrows():
        gt_bbox = row['seg_ans']
        if gt_bbox == 'nan':
            gt = 'not visible'
        else:
            center_x = gt_bbox[0]
            center_y = gt_bbox[1]
            if center_x < 0.45 and center_y < 0.45:
                gt = 'upper left'
            elif center_x >= 0.45 and center_x < 0.55 and center_y < 0.45:
                gt = 'upper center'
            elif center_x >= 0.55 and center_y < 0.45:
                gt = 'upper right'
            elif center_x < 0.45 and center_y >= 0.45 and center_y < 0.55:
                gt = 'middle left'
            elif center_x >= 0.45 and center_x < 0.55 and center_y >= 0.45 and center_y < 0.55:
                gt = 'center'
            elif center_x >= 0.55 and center_x < 0.55 and center_y >= 0.45 and center_y < 0.55:
                gt = 'middle right'
            elif center_x < 0.45 and center_y >= 0.55:
                gt = 'lower left'
            elif center_x >= 0.45 and center_x < 0.55 and center_y >= 0.55:
                gt = 'lower center'
            elif center_x >= 0.55 and center_y >= 0.55:
                gt = 'lower right'
        
        
        if row['prediction'] == 'failed':
            failed_items += 1
            continue
        else:
            gt_list.append(gt)
            pred_list.append(row['prediction'])
    
    total_samples = len(data) - failed_items
    accuracy = sum(np.array(gt_list) == np.array(pred_list)) / total_samples
    precision = precision_score(gt_list, pred_list, average='macro')  # 多分类用macro平均
    recall = recall_score(gt_list, pred_list, average='macro')
    f1 = f1_score(gt_list, pred_list, average='macro')


class SegDataset(ImageBaseDataset):
    TYPE = 'PENDING'
    MODALITY = 'IMAGE'

    TSV_PATH = {
        "put TSV here"
    }

    def build_prompt(self, line, task_type):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['img_path'])
        else:
            tgt_path = self.dump_image(line)
        anatomy = line['anatomy_location']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None

        prompt = ''
        if hint is not None:
            prompt += f'{hint}.\n'
        
        if self.dataset_name in ['04', '17', '23', '32', '64']:
            prompt += f"You are a radiologist analyzing an ultrasound image about {anatomy}.\n"
            prompt += "Your task is to identify the primary location of any visible lesion(s) relative to the boundaries of the displayed image. Consider the lesion's center location or most prominent area when deciding. Choose the single option from the list below that best describes this location, even if the fit is approximate.\n"
            prompt += "Choose the single most appropriate location from the following list: upper left, upper center, upper right, middle left, center, middle right, lower left, lower center, lower right, not visible.\n"
            prompt += "Output format: only one or two word(s) representing the chosen location. No additional text or formatting is allowed."
        elif self.dataset_name == '48':
            keypoint_str = line['keypoints']
            # 处理keypoint_str可能是浮点数的情况
            if isinstance(keypoint_str, float):
                # 如果是浮点数，使用默认值
                kp = "heart valve location"
            else:
                try:
                    # 尝试解析JSON
                    kp_dict = json.loads(keypoint_str)
                    kp = list(kp_dict.keys())[0]
                except (json.JSONDecodeError, TypeError):
                    # 如果解析失败，使用默认值
                    kp = "heart valve location"
            
            prompt += "You are a radiologist analyzing an ultrasound image of the heart.\n"
            prompt += f"Your task is to determine the {kp}.\n"
            prompt += "Choose the single most appropriate location from the following list: upper left, upper center, upper right, middle left, center, middle right, lower left, lower center, lower right, not visible.\n"
            prompt += "Output format: only one or two word(s) representing the chosen location. No additional text or formatting is allowed."
        else:
            seg_channel_name = line['seg_channel_name']
            prompt += f"You are a radiologist analyzing an ultrasound image about {anatomy}.\n"
            prompt += f"Your task is to determine the primary location, relative to the image boundaries, for each visible structure listed in {seg_channel_name}.\n"
            prompt += "*   Consider the structure's center or most prominent area when deciding its location.\n"
            prompt += "*   Choose the single option from the list below that best describes the location, even if the fit is approximate.\n"
            prompt += "Location Options: upper left, upper center, upper right, middle left, center, middle right, lower left, lower center, lower right, not visible.\n"
            prompt += "Output format: only one or two word(s) representing the chosen location. No additional text or formatting is allowed."


        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, task_type = 'cla', **judge_kwargs):
        sys.path.append('../VLMEvalKit/')
        from eval_with_xlsx import seg_eval
        data = load(eval_file)
        data = data.sort_values(by='index')
        if task_type == 'seg':
            mean_iou_dict = seg_eval(data, eval_file)
            return mean_iou_dict
        else:
            raise NameError('This task type should be segmentation/detection')
        
