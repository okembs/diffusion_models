import torch 
import torch.nn as nn
import os 
from pathlib import Path
import argparse
from  Dppmclassifier import DDPM
from diffusion import LatentDiffusionWrapper
from labml_nn.diffusion.stable_diffusion.sampler.ddpm import DDPMSampler

# the dppm sampler will be used for generating images that are already trained by the model 
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
#N_steps : is the number of sampling steps
class Text2image: 
    model : LatentDiffusionWrapper
    def __init__(self , checkpoint:Path , sampler_name , N_steps) : 
        
        model = torch.load(checkpoint , weights_only=False).to(device)
        self.N_steps = N_steps
        if sampler_name == 'ddpm' : 
            self.sampler = DDPM(model , N_steps , device)
        else : 
            print(f'try using another one')
     # h : is the height of the image 
     # w : is the width of the image
     #dest_path : is the destination part  for storing the image
     # batch_size : is the batch size of the image
     # uncond scale: is the unconditional guidance scale
     # c : is the number of channels
     # f : is the latent space to resolution

    @torch.no_grad()
    def call(self , dest_path:str ,  prompt:str  ,batch_size:int = 3 , h:int = 512 , w:int = 512 , undcond_scale:float = 52 ) : 
        self.destination = dest_path
        self.prompt = prompt
        self.batch_size = batch_size

        c = 4 
        f = 8
        prompts = prompt[self.batch_size] * self.prompt
        with torch.cuda.amp.autocast('cuda') : 
            # if the uncond scale is not 1 , get empty condition embeddings
            if undcond_scale != 1 : 
                un_cond = self.model.get_text_embedding( batch_size * [""])
            else : 
                un_cond = None 
            # sample and get the latent space
        x = self.sampler
                


        