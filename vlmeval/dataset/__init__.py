import warnings

from .image_base_0302 import ImageBaseDataset, img_root_map
from .image_seg import SegDataset
from .image_cla import ClaDataset
from .image_report import ReportDataset
from .image_measurement import MeasureDataset
# from .image_base import img_root_map, ImageBaseDataset
from .image_caption import ImageCaptionDataset
from .image_yorn import ImageYORNDataset
from .image_mcq import (
    ImageMCQDataset, MMMUDataset, CustomMCQDataset, MUIRDataset, GMAIMMBenchDataset, MMERealWorld, HRBenchDataset,
    NaturalBenchDataset
)
from .image_mt import MMDUDataset
from .image_vqa import (
    ImageVQADataset, MathVision, OCRBench, MathVista, LLaVABench, MMVet, MTVQADataset, TableVQABench,
    CustomVQADataset, CRPE, MathVerse, OlympiadBench, QSpatial, VizWiz, MMNIAH, WeMath, LogicVista
)

from .image_ccocr import CCOCRDataset
from .text_mcq import CustomTextMCQDataset, TextMCQDataset

from .vcr import VCRDataset
from .mmlongbench import MMLongBench
from .dude import DUDE
from .slidevqa import SlideVQA
from .vl_rewardbench import VLRewardBench

from .mmbench_video import MMBenchVideo
from .videomme import VideoMME
from .mvbench import MVBench, MVBench_MP4
from .mlvu import MLVU, MLVU_MCQ, MLVU_OpenEnded
from .tempcompass import TempCompass, TempCompass_Captioning, TempCompass_MCQ, TempCompass_YorN
from .longvideobench import LongVideoBench
from .video_concat_dataset import ConcatVideoDataset
from .mmgenbench import MMGenBench
from .cgbench import CGBench_MCQ_Grounding_Mini, CGBench_OpenEnded_Mini, CGBench_MCQ_Grounding, CGBench_OpenEnded

from .miabench import MIABench
from .cmmmu import CMMMU
from .wildvision import WildVision
from .mmmath import MMMath
from .dynamath import Dynamath
from .utils import *
from .video_dataset_config import *
from ..smp import *
import re


class ConcatDataset(ImageBaseDataset):
    # This dataset takes multiple dataset names as input and aggregate them into a single dataset.
    # Each single dataset should not have a field named `SUB_DATASET`

    DATASET_SETS = {
        'MMMB': ['MMMB_ar', 'MMMB_cn', 'MMMB_en', 'MMMB_pt', 'MMMB_ru', 'MMMB_tr'],
        'MTL_MMBench_DEV': [
            'MMBench_dev_ar', 'MMBench_dev_cn', 'MMBench_dev_en',
            'MMBench_dev_pt', 'MMBench_dev_ru', 'MMBench_dev_tr'
        ]
    }

    def __init__(self, dataset):
        datasets = self.DATASET_SETS[dataset]
        self.dataset_map = {}
        # The name of the compliation
        self.dataset_name = dataset
        self.datasets = datasets
        for dname in datasets:
            dataset = build_dataset(dname)
            assert dataset is not None, dataset
            self.dataset_map[dname] = dataset
        TYPES = [x.TYPE for x in self.dataset_map.values()]
        MODALITIES = [x.MODALITY for x in self.dataset_map.values()]
        assert np.all([x == TYPES[0] for x in TYPES]), (datasets, TYPES)
        assert np.all([x == MODALITIES[0] for x in MODALITIES]), (datasets, MODALITIES)
        self.TYPE = TYPES[0]
        self.MODALITY = MODALITIES[0]
        data_all = []
        for dname in datasets:
            data = self.dataset_map[dname].data
            data['SUB_DATASET'] = [dname] * len(data)
            data_new = localize_df(data, dname, nproc=16)
            data_all.append(data_new)

        data = pd.concat(data_all)
        data['original_index'] = data.pop('index')
        data['index'] = np.arange(len(data))
        self.data = data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        idx = line['original_index']
        dname = line['SUB_DATASET']
        org_data = self.dataset_map[dname].data
        org_line = cp.deepcopy(org_data[org_data['index'] == idx]).iloc[0]
        return self.dataset_map[dname].build_prompt(org_line)

    def dump_image(self, line):
        # Assert all images are pre-dumped
        assert 'image' not in line
        assert 'image_path' in line
        tgt_path = toliststr(line['image_path'])
        return tgt_path

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_SETS)

    def evaluate(self, eval_file, **judge_kwargs):
        suffix = eval_file.split('.')[-1]
        # First, split the eval_file by dataset
        data_all = load(eval_file)
        for dname in self.datasets:
            tgt = eval_file.replace(self.dataset_name, dname)
            data_sub = data_all[data_all['SUB_DATASET'] == dname]
            data_sub.pop('index')
            data_sub['index'] = data_sub.pop('original_index')
            data_sub.pop('SUB_DATASET')
            dump(data_sub, tgt)
        # Then, evaluate each dataset separately
        results_all = []
        for dname in self.datasets:
            tgt = eval_file.replace(self.dataset_name, dname)
            res = self.dataset_map[dname].evaluate(tgt, **judge_kwargs)
            assert isinstance(res, pd.DataFrame)
            res['DATASET'] = [dname] * len(res)
            results_all.append(res)
        result = pd.concat(results_all)
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(result, score_file)
        return result


# Add new supported dataset class here
IMAGE_DATASET = [
    ImageCaptionDataset, ImageYORNDataset, ImageMCQDataset, ImageVQADataset, MathVision,
    MMMUDataset, OCRBench, MathVista, LLaVABench, MMVet, MTVQADataset, TableVQABench,
    MMLongBench, VCRDataset, MMDUDataset, DUDE, SlideVQA, MUIRDataset, CCOCRDataset,
    GMAIMMBenchDataset, MMERealWorld, HRBenchDataset, CRPE, MathVerse, NaturalBenchDataset,
    MIABench, OlympiadBench, WildVision, MMMath, QSpatial, Dynamath, MMGenBench, VizWiz, MMNIAH,
    CMMMU, VLRewardBench, WeMath, LogicVista,
    # SegDataset, ClaDataset
]

VIDEO_DATASET = [
    MMBenchVideo, VideoMME, MVBench, MVBench_MP4, LongVideoBench,
    MLVU, MLVU_MCQ, MLVU_OpenEnded,
    TempCompass, TempCompass_MCQ, TempCompass_Captioning, TempCompass_YorN,
    CGBench_MCQ_Grounding_Mini, CGBench_OpenEnded_Mini, CGBench_MCQ_Grounding, CGBench_OpenEnded
]

TEXT_DATASET = [
    TextMCQDataset
]

CUSTOM_DATASET = [
    CustomMCQDataset, CustomVQADataset, CustomTextMCQDataset
]

DATASET_COLLECTION = [ConcatDataset, ConcatVideoDataset]

DATASET_CLASSES = IMAGE_DATASET + VIDEO_DATASET + TEXT_DATASET + CUSTOM_DATASET + DATASET_COLLECTION
SUPPORTED_DATASETS = []
for DATASET_CLS in DATASET_CLASSES:
    SUPPORTED_DATASETS.extend(DATASET_CLS.supported_datasets())


def DATASET_TYPE(dataset, *, default: str = 'MCQ') -> str:
    for cls in DATASET_CLASSES:
        if dataset in cls.supported_datasets():
            if hasattr(cls, 'TYPE'):
                return cls.TYPE
    # Have to add specific routine to handle ConcatDataset
    if dataset in ConcatDataset.DATASET_SETS:
        dataset_list = ConcatDataset.DATASET_SETS[dataset]
        TYPES = [DATASET_TYPE(dname) for dname in dataset_list]
        assert np.all([x == TYPES[0] for x in TYPES]), (dataset_list, TYPES)
        return TYPES[0]

    if 'openended' in dataset.lower():
        return 'VQA'
    warnings.warn(f'Dataset {dataset} is a custom one and not annotated as `openended`, will treat as {default}. ')
    return default


def DATASET_MODALITY(dataset, *, default: str = 'IMAGE') -> str:
    if dataset is None:
        warnings.warn(f'Dataset is not specified, will treat modality as {default}. ')
        return default
    for cls in DATASET_CLASSES:
        if dataset in cls.supported_datasets():
            if hasattr(cls, 'MODALITY'):
                return cls.MODALITY
    # Have to add specific routine to handle ConcatDataset
    if dataset in ConcatDataset.DATASET_SETS:
        dataset_list = ConcatDataset.DATASET_SETS[dataset]
        MODALITIES = [DATASET_MODALITY(dname) for dname in dataset_list]
        assert np.all([x == MODALITIES[0] for x in MODALITIES]), (dataset_list, MODALITIES)
        return MODALITIES[0]

    if 'VIDEO' in dataset.lower():
        return 'VIDEO'
    elif 'IMAGE' in dataset.lower():
        return 'IMAGE'
    warnings.warn(f'Dataset {dataset} is a custom one, will treat modality as {default}. ')
    return default


def build_dataset(dataset_name, task_type='cla', **kwargs):
    for cls in DATASET_CLASSES:
        if dataset_name in supported_video_datasets:
            return supported_video_datasets[dataset_name](**kwargs)
        elif dataset_name in cls.supported_datasets():
            return cls(dataset=dataset_name, **kwargs)

    warnings.warn(f'Dataset {dataset_name} is not officially supported. ')

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

        '48': '/media/ps/data-ssd/json_processing/ale_tsv_output/single_keypoint/48.tsv',
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
        '28_class': '/media/ps/data-ssd/json_processing/ale_tsv_output/28.Knee_Classification.tsv',

        '18': '/media/ps/data-ssd/json_processing/ale_tsv_output/18_1.Ultrasound_Heart_Segmentation_Dataset.tsv',
        '27': '/media/ps/data-ssd/json_processing/ale_tsv_output/27.tsv',
        '31': '/media/ps/data-ssd/json_processing/ale_tsv_output/31.tsv',
        '50': '/media/ps/data-ssd/json_processing/ale_tsv_output/50.tsv',
        '57': '/media/ps/data-ssd/json_processing/ale_tsv_output/57_1.Liver_Ultrasound_Segmentation_Dataset.tsv',

        '10': '/media/ps/data-ssd/json_processing/ale_tsv_output/10.tsv',
        '11': '/media/ps/data-ssd/json_processing/ale_tsv_output/11.Thyroid_US_Images.tsv',
        '39': '/media/ps/data-ssd/json_processing/ale_tsv_output/39_translated.tsv',
    }

    tsv_ROOT = "/media/ps/data-ssd/benchmark/VLMEvalKit/dataset_tsv/04-14-lhn/"
    if dataset_name == 'BreastDataset':
        data_file = osp.join(tsv_ROOT, '23.breast-ultrasound-images-dataset.tsv')
    elif dataset_name == 'FetalDataset':
        data_file = osp.join(tsv_ROOT, '03.FETAL.tsv')
    else:
        if dataset_name in TSV_PATH:
            key_to_use = dataset_name
        else:
            match = re.match(r'^\d+', dataset_name)
            key_to_use = match.group(0)
        data_file = TSV_PATH[key_to_use]
    # else:
    #     data_file = osp.join(LMUDataRoot(), f'{dataset_name}.tsv')
    
    print(f"🤖 TSV: {data_file}")
    if not osp.exists(data_file):
        warnings.warn(f'Data file {data_file} does not exist. Dataset building failed. ')
        return None

    data = load(data_file)
    if task_type == 'cla':
        return ClaDataset(dataset=dataset_name, **kwargs)
    
    if task_type == 'seg':
        return SegDataset(dataset=dataset_name, **kwargs)
    
    if task_type == 'report':
        return ReportDataset(dataset=dataset_name, **kwargs)

    if task_type == 'measurement':
        return MeasureDataset(dataset=dataset_name, **kwargs)
 
    # if 'question' not in [x.lower() for x in data.columns]:
    #     warnings.warn(f'Data file {data_file} does not have a `question` column. Dataset building failed. ')
    #     return None

    if 'A' in data and 'B' in data:
        if 'image' in data or 'image_path' in data:
            warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom MCQ dataset. ')
            return CustomMCQDataset(dataset=dataset_name, **kwargs)
        else:
            warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom Text MCQ dataset. ')
            return CustomTextMCQDataset(dataset=dataset_name, **kwargs)
    else:
        warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom VQA dataset. ')
        return CustomVQADataset(dataset=dataset_name, **kwargs)


__all__ = [
    'build_dataset', 'img_root_map', 'build_judge', 'extract_answer_from_item', 'prefetch_answer', 'DEBUG_MESSAGE'
] + [cls.__name__ for cls in DATASET_CLASSES]
