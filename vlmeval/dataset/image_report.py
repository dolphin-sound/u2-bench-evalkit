from .image_base_0302 import ImageBaseDataset
import pandas as pd
from .utils import build_judge, DEBUG_MESSAGE, extract_json_data
from ..smp import *
import ast
import re
import random
import json
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from bert_score import score as bert_score
# from sentence_transformers import SentenceTransformer


class MedicalReportScorer:
    def __init__(self, refs, hyps):
        """
        refs: { 'case001': ["专家报告1文本", "专家报告2文本"], ... } 
        hyps: { 'case001': "生成报告文本", ... }
        """
        self.refs = refs
        self.hyps = hyps
        self.scorers = [
            (Bleu(4), 'Bleu'),
            (Rouge(), 'Rouge'),
            # (Cider(), 'Cider')
        ]
        
    def evaluate(self):
        metrics = {}
        # COCO官方指标计算
        for scorer, name in self.scorers:
            score, _ = scorer.compute_score(self.refs, self.hyps)
            if name == 'Bleu':
                metrics.update({f'Bleu-{i+1}': v*100 for i, v in enumerate(score)})
            else:
                metrics[name] = score * 100
                
        # BERTScore
        # sentence
        # refs_list = [' '.join(refs) for refs in self.refs.values()]
        # hyps_list = list(self.hyps.values())
        # all_sentence = refs_list + hyps_list
        # model = SentenceTransformer("all-MiniLM-L6-v2")
        # embeddings = model.encode(all_sentence)
        # similarities = model.similarity(embeddings, embeddings)
        
        return metrics
    

class ReportDataset(ImageBaseDataset):
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
            
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'

        prompt += f"You are a radiologist analyzing an ultrasound image focused on the {anatomy}."
        if self.dataset_name == '39':
            prompt += f"Your task is generate a concise and informative radiological report based strictly on the visual findings within the provided image. Your report should describe the primary organ's appearance (size, shape, borders/capsule), its parenchymal echotexture (e.g., homogeneous, heterogeneous, echogenicity relative to reference structures), and identify any visible abnormalities (e.g., masses, cysts, fluid collections, calcifications, ductal dilation). Comment on relevant adjacent structures if visualized. Use standard radiological terminology."
            prompt += "Output format: Strings, that is your report."
            prompt += "Example: The liver morphology is full with a smooth capsule. The parenchymal echotexture is fine and diffusely increased. Visualization of the portal venous system is suboptimal. Intrahepatic and extrahepatic bile ducts are not dilated. The main portal vein diameter is within normal limits. The gallbladder is normal in size and shape. The wall is smooth and not thickened. No obvious abnormal echoes are seen within the lumen. The pancreas is normal in size and shape with homogeneous parenchymal echotexture. The pancreatic duct is not dilated. No definite space-occupying lesion is seen within the pancreas. The spleen is normal in size and shape with homogeneous parenchymal echotexture. No obvious space-occupying lesion is seen within the spleen."
        else:
            prompt += f"Your task is to generate a concise and informative caption that accurately describes the key anatomical structures and any significant findings visible in the provided ultrasound image."
            prompt += "Output format: A single string constituting the image caption. Output only the generated caption text itself. Do not include any introductory phrases (like \"Caption:\"), labels, explanations, or additional formatting."
            prompt += "Examples:"
            if self.dataset_name == '10':
                prompt += "Example1: Fetal phantom ultrasound image showing standard diagnostic plane for abdominal circumference (AC) measurement\n" 
                prompt += "Example2: Fetal phantom ultrasound image showing standard diagnostic plane for biparietal diameter (BPD) measurement\n"
                prompt += "Example3: Fetal phantom ultrasound image showing standard diagnostic plane for femur length (FL) measurement\n"
            elif self.dataset_name == '11':
                prompt += "Example1: Thyroid nodule in the right lobe. TI-RADS level 3, Benign.\n"
                prompt += "Example2: Thyroid nodule in the left lobe. TI-RADS level 3, Benign.\n"
                prompt += "Example3: Thyroid nodule in the right lobe. TI-RADS level 4, Benign.\n"
            elif self.dataset_name == '44':
                prompt += "Example1: no single B-lines, B-lines are fused together into the picture of a white lung\n"
                prompt += "Example2: liver on the right side\n"
                prompt += "Example3: white lung? "

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

        refs = {}
        for i, row in data.iterrows():
            if 'gt_report' in data.columns and row['gt_report']:
                refs[row['index']] = [row['gt_report']]
            else:
                refs[row['index']] = [row['caption']]
        hyps = {row['index']: [row['prediction']] for _, row in data.iterrows()}

        medical_scorer = MedicalReportScorer(refs, hyps)
        report_dict = medical_scorer.evaluate()
        
        txt_file = eval_file.replace('.xlsx', '.txt')
        with open(txt_file, 'w') as tf:
            tf.write(str(report_dict))
        return report_dict
       