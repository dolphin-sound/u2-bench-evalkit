import torch
import sys
import os.path as osp
import warnings
from transformers import StoppingCriteriaList
from .base import BaseModel
from PIL import Image
# from rich import print

class MiniGPT4(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self,
                 mode='v2',
                 root='/media/ps/data-ssd/benchmark/VLMEvalKit/vlmeval/vlm/MiniGPT4_lib',
                 temperature=1,
                 max_out_len=1000,
                 img_size=448,):

        if root is None:
            warnings.warn(
                'Please set root to the directory of MiniGPT-4, which is cloned from here: '
                'https://github.com/Vision-CAIR/MiniGPT-4. '
            )

        # breakpoint()
        if mode == 'v2':
            cfg = 'minigptv2_eval.yaml'
        elif mode == 'v1_7b':
            cfg = 'minigpt4_7b_eval.yaml'
        elif mode == 'v1_13b':
            cfg = 'minigpt4_13b_eval.yaml'
        else:
            raise NotImplementedError

        self.mode = mode
        self.temperature = temperature
        self.max_out_len = max_out_len
        self.root = root
        this_dir = osp.dirname(__file__)

        self.cfg = osp.join(this_dir, 'misc', cfg)
        sys.path.append(self.root)
        # breakpoint()
        from omegaconf import OmegaConf
        from minigpt4.common.registry import registry
        from minigpt4.conversation.conversation import Conversation, SeparatorStyle

        device = torch.cuda.current_device()
        # self.device = device

        cfg_path = self.cfg
        cfg = OmegaConf.load(cfg_path)

        model_config = cfg.model
        model_config.device_8bit = device
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(device)
        vis_processor_cfg = cfg.datasets.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        model = model.eval()
        self.model = model
        self.vis_processor = vis_processor
        # self.CONV_VISION = CONV_VISION_minigptv2 if self.mode == 'v2' else CONV_VISION_Vicuna0
        self.CONV_VISION = Conversation(
            system="",
            roles=(r"<s>[INST] ", r" [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )
        # stop_words_ids = [[835], [2277, 29937]]
        # stop_words_ids = [torch.tensor(ids).to(device) for ids in stop_words_ids]
        # self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def generate_inner(self, message, dataset=None):
        sys.path.append(self.root)
        from minigpt4.conversation.conversation import Chat
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        device = torch.cuda.current_device()

        # 初始化 Chat 对象
        chat = Chat(self.model, self.vis_processor, device=device)

        # 初始化对话状态
        chat_state = self.CONV_VISION.copy()
        img_list = []

        # 加载和处理图像
        try:
            image = Image.open(image_path).convert('RGB')
            print(f"Loaded image: {image_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return "Error loading image"

        # 上传图像并编码
        llm_message = chat.upload_img(image, chat_state, img_list)
        # print(f"llm_message: {llm_message}")
        chat.encode_img(img_list)
        # print(f"Encoded images: {img_list}")

        # 处理用户输入的 prompt
        answer = "[vqa]"
        answer += prompt
        chat.ask(answer, chat_state)
        print(f"Answer: {answer}")

        # 生成模型响应
        with torch.inference_mode():
            msg = chat.answer(conv=chat_state, 
                              img_list=img_list,
                              temperature=self.temperature,
                              max_length=self.max_out_len)[0]
        print(f"🤖 MiniGPT-med says: {msg}")
        # print(f"chat_state after answer: {chat_state}")
        # breakpoint()
        return msg
