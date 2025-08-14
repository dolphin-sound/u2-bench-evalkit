import torch
from PIL import Image
import os.path as osp
import sys
from .base import BaseModel
from ..smp import *


class InstructBLIP(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, name):
        self.config_map = {
            'instructblip_7b': 'misc/blip2_instruct_vicuna7b.yaml',
            'instructblip_13b': 'misc/blip2_instruct_vicuna13b.yaml',
        }
        sys.path.append('/media/ps/data-ssd/benchmark/VLMEvalKit/LAVIS')
        self.file_path = __file__
        config_root = osp.dirname(self.file_path)

        try:
            from lavis.models import load_preprocess, load_model_and_preprocess
            from omegaconf import OmegaConf
            from lavis.common.registry import registry
        except Exception as e:
            logging.critical('Please install lavis before using InstructBLIP. ')
            raise e

        assert name in self.config_map
        cfg_path = osp.join(config_root, self.config_map[name])
        cfg = OmegaConf.load(cfg_path)

        # model_cfg = cfg.model
        # assert osp.exists(model_cfg.llm_model) or splitlen(model_cfg.llm_model) == 2
        # model_cls = registry.get_model_class(name='blip2_vicuna_instruct')
        # model = model_cls.from_config(model_cfg)
        # model.eval()

        model_type = "vicuna7b" if '7b' in name else "vicuna13b"
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        device = self.device
        model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type=model_type, is_eval=True, device=device)
        model.to(device)
        self.model = model
        self.kwargs = {'max_length': 1024}

        preprocess_cfg = cfg.preprocess
        vis_processors, _ = load_preprocess(preprocess_cfg)
        self.vis_processors = vis_processors
        # self.txt_processors = txt_processors

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        vis_processors = self.vis_processors
        raw_image = Image.open(image_path).convert('RGB')
        image_tensor = vis_processors['eval'](raw_image).unsqueeze(0).to(self.device)
        # prompt = self.txt_processors['eval'](prompt).to(self.device)
        outputs = self.model.generate(dict(image=image_tensor, prompt=prompt)) 
        print(f"🪄Instruct-blip says: {outputs[0]}")
        return outputs[0]
