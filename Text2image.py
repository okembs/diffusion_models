import torch 
import torch.nn as nn
import os 
from pathlib import Path
import argparse
from  Dppmclassifier import DDPM
from diffusion import LatentDiffusionWrapper


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
#N_steps : is the number of sampling steps
class Text2image: 
    model : LatentDiffusionWrapper
    def __init__(self , checkpoint:Path , sampler_name , N_steps) : 
        
        model = torch.load(Path , weights_only=False).to(device)
        self.N_steps = N_steps
        if sampler_name == 'ddpm' : 
            self.sampler = DDPM(model , N_steps , device)
        else : 
            print(f'try using another one')
     # h : is the height of the image 
     # w : is the width of the image
     #dest_path : is the destination part  for storing the image
     # batch_size : is the batch size of the image

    @torch.no_grad()
    def call(self , dest_path:str ,  prompt:str  ,batch_size:int = 3 , h:int = 512 , w:int = 512 ,  ) : 
        self.desti
        pass

