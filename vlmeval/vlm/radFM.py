import warnings
from .base import BaseModel
from ..smp import *
import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from torchvision import transforms
from PIL import Image, ImageFile 
from rich.console import Console
from rich.theme import Theme
from rich import print
import accelerate
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


def get_tokenizer(tokenizer_path, max_img_size = 100, image_num = 32):
    '''
    Initialize the image special tokens
    max_img_size denotes the max image put length and image_num denotes how many patch embeddings the image will be encoded to 
    '''
    if isinstance(tokenizer_path,str):
        image_padding_tokens = []
        text_tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path,
        )
        special_token = {"additional_special_tokens": ["<image>","</image>"]}
        for i in range(max_img_size):
            image_padding_token = ""
            
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append("<image"+str(i*image_num+j)+">")
            image_padding_tokens.append(image_padding_token)
            text_tokenizer.add_special_tokens(
                special_token
            )
            ## make sure the bos eos pad tokens are correct for LLaMA-like models
            text_tokenizer.pad_token_id = 0
            text_tokenizer.bos_token_id = 1
            text_tokenizer.eos_token_id = 2    
    
    return  text_tokenizer,image_padding_tokens    

def combine_and_preprocess(question,image_list,image_padding_tokens):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512,512],scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
    images  = []
    new_qestions = [_ for _ in question]
    padding_index = 0
    for img in image_list:
        img_path = img['img_path']
        position = img['position']
        image = Image.open(img_path).convert('RGB')   
        image = transform(image)
        image = image.unsqueeze(0).unsqueeze(-1) # c,w,h,d
        
        ## pre-process the img first
        target_H = 512 
        target_W = 512 
        target_D = 4 
        # This can be different for 3D and 2D images. For demonstration we here set this as the default sizes for 2D images. 
        images.append(torch.nn.functional.interpolate(image, size = (target_H,target_W,target_D)))
        
        ## add img placeholder to text
        new_qestions[position] = "<image>"+ image_padding_tokens[padding_index] +"</image>" + new_qestions[position]
        padding_index +=1
    
    vision_x = torch.cat(images,dim = 1).unsqueeze(0) #cat tensors and expand the batch_size dim
    text = ''.join(new_qestions) 
    return text, vision_x,


class RadFM(BaseModel):
    def __init__(self, **kwargs):
        custom_theme = Theme({
            "info": "dim cyan",
            "warning": "magenta",
            "danger": "bold red"
        })
        console = Console(theme=custom_theme)
        import sys
        sys.path.append('/media/ps/data-ssd/benchmark/VLMEvalKit/RadFM/Quick_demo')
        from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
        console.print("[bold red][INFO][/bold red] Setup tokenizer", style='info')
        self.text_tokenizer, self.image_padding_tokens = get_tokenizer('/media/ps/data-ssd/benchmark/VLMEvalKit/RadFM/Quick_demo/Language_files')
        console.print("[bold red][INFO][/bold red] Finish loading tokenizer", style='info')

        console.print("[bold red][INFO][/bold red] Setup Model", style='info')
        with init_empty_weights():
            model = MultiLLaMAForCausalLM(
                lang_model_path='/media/ps/data-ssd/benchmark/VLMEvalKit/RadFM/Quick_demo/Language_files', ### Build up model based on LLaMa-13B config
            )
        model = load_checkpoint_and_dispatch(model, checkpoint='/media/ps/data-ssd/benchmark/VLMEvalKit/RadFM/Quick_demo/pytorch_model.bin', device_map ='auto') # Please dowloud our checkpoint from huggingface and Decompress the original zip file first
        model.eval() 
        console.print("[bold red][INFO][/bold red] Finish loading model", style='info')

        self.model = model

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = [
            {
                'img_path': image_path,
                'position': 0, #indicate where to put the images in the text string, range from [0,len(question)-1]
            }
        ]
        text, vision_x = combine_and_preprocess(prompt, image, self.image_padding_tokens)
        with torch.no_grad():
            lang_x = self.text_tokenizer(
                    text, max_length=2048, truncation=True, return_tensors="pt"
            )['input_ids']
            
            vision_x = vision_x
            generation = self.model.generate(lang_x, vision_x)
            generated_texts = self.text_tokenizer.batch_decode(generation, skip_special_tokens=True) 
            print('---------------------------------------------------')
            print('[magenta]Input[/magenta]: ', prompt)
            print('[magenta]Output[/magenta]: ', generated_texts[0])
        return generated_texts[0]