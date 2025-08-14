import pandas as pd
from abc import abstractmethod
from ..smp import *
import os
import hashlib


def img_root_map(dataset):
    if 'MM_NIAH' in dataset:
        return 'MMNIAH'
    if 'CRPE' in dataset:
        return 'CRPE'
    if 'OCRVQA' in dataset:
        return 'OCRVQA'
    if 'COCO_VAL' == dataset:
        return 'COCO'
    if 'MMMU' in dataset:
        return 'MMMU'
    if "QSpatial" in dataset:
        return "QSpatial"
    if "Breast" in dataset:
        return "BreastDataset"
    if "Fetal" in dataset:
        return "FetalDataset"

    mmbench_root_map = {
        'MMBench_DEV_EN': 'MMBench', 'MMBench_TEST_EN': 'MMBench',
        'MMBench_DEV_CN': 'MMBench', 'MMBench_TEST_CN': 'MMBench',
        'MMBench': 'MMBench', 'MMBench_CN': 'MMBench',
        'MMBench_DEV_EN_V11': 'MMBench_V11', 'MMBench_TEST_EN_V11': 'MMBench_V11',
        'MMBench_DEV_CN_V11': 'MMBench_V11', 'MMBench_TEST_CN_V11': 'MMBench_V11',
        'MMBench_V11': 'MMBench', 'MMBench_CN_V11': 'MMBench',
    }
    if dataset in mmbench_root_map:
        return mmbench_root_map[dataset]
    return dataset


class ImageBaseDataset:
    TYPE = 'PENDING'
    MODALITY = 'IMAGE'
    # DATASET_URL = {}
    # DATASET_MD5 = {}
    TSV_PATH = {
        "put TSV here"
    }

    def __init__(self, dataset='MMBench', skip_noimg=True):
        ROOT = LMUDataRoot()
        # You can override this variable to save image files to a different directory
        self.dataset_name = dataset
        self.img_root = osp.join(ROOT, img_root_map(dataset))
        data = self.load_data(dataset)
        self.skip_noimg = skip_noimg
        if skip_noimg and 'image' in data:
            data = data[~pd.isna(data['image'])]

        # data['index'] = [str(x) for x in data['index']]
        if 'index' in data.columns:
            data['index'] = data['index'].astype(str)
        else:
            data.reset_index(inplace=True)  # 使用默认索引
            data.rename(columns={'index': 'original_index'}, inplace=True)
            data['index'] = data['original_index'].astype(str)

        self.meta_only = True

        # The image field can store the base64 encoded image or another question index (for saving space)
        if 'img_data' in data:
            data['img_data'] = [str(x) for x in data['img_data']]
            image_map = {x: y for x, y in zip(data['index'], data['img_data'])}
            for k in image_map:
                if len(image_map[k]) <= 64:
                    idx = image_map[k]
                    assert idx in image_map and len(image_map[idx]) > 64
                    image_map[k] = image_map[idx]

            images = [toliststr(image_map[k]) for k in data['index']]
            data['img_data'] = [x[0] if len(x) == 1 else x for x in images]
            self.meta_only = False

        if 'img_path' in data:
            paths = [toliststr(x) for x in data['img_path']]
            data['img_path'] = [x[0] if len(x) == 1 else x for x in paths]

        if np.all([istype(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]

        self.data = data
        self.post_build(dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(self.data.iloc[idx])

    def prepare_tsv(self, tsv_path):
        return load(tsv_path, fmt='tsv')

    def dump_image(self, line):
        """将base64图像数据解码并保存为文件"""
        # 修改保存路径到用户home目录
        home_dir = os.path.expanduser('~')
        cache_dir = os.path.join(home_dir, '.cache', 'vlmeval', 'images')
        os.makedirs(cache_dir, exist_ok=True)
        
        # 使用哈希值作为文件名，避免冲突
        img_hash = hashlib.md5(line['img_data'].encode()).hexdigest()
        tgt_path = os.path.join(cache_dir, f"{img_hash}.jpg")
        
        if not os.path.exists(tgt_path):
            decode_base64_to_image_file(line['img_data'], tgt_path)
        return tgt_path

    def display(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        assert isinstance(line, pd.Series) or isinstance(line, dict)
        mmqa_display(line)

    # Return a list of dataset names that are supported by this class, can override
    @classmethod
    def supported_datasets(cls):
        return list(cls.TSV_PATH)

    # Given the dataset name, return the dataset as a pandas dataframe, can override
    def load_data(self, dataset):
        # url = self.DATASET_URL[dataset]
        # file_md5 = self.DATASET_MD5[dataset] if dataset in self.DATASET_MD5 else None
        tsv_path = self.TSV_PATH[dataset]
        return self.prepare_tsv(tsv_path)

    # Post built hook, will be called after the dataset is built, can override
    def post_build(self, dataset):
        pass

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line, task_type):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['img_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    # Given the prediction file, return the evaluation results in the format of a dictionary or pandas dataframe
    @abstractmethod
    def evaluate(self, eval_file, **judge_kwargs):
        pass

    def get_tsv_file(self, dataset):
        return self.TSV_PATH[dataset]
