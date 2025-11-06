import torch 
import  torch.nn as nn 
from torch.nn import functional as F 
from typing import List , Optional
from diffusion import LatentDiffusionWrapper

class DiffusionSampler : 
    model: LatentDiffusionWrapper 
    def __init__(self , model:LatentDiffusionWrapper):
        super().__init__()
        self.model = model 
        # get the number of steps trained for the model
        self.n_steps = self.model.n_steps

    def get_eps(self , x:torch.Tensor , t:torch.Tensor, c:torch.Tensor , uncond_scale:float , uncond_cond) : 
        if uncond_cond is None and uncond_scale == 1 : 
            return self.model(x , t , c)
        
        # duplicate xt  and t 
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([x] * 2)

        # cocantenated c and u 
        c_in = torch.cat([uncond_cond , c])

        e_t_uncond , e_t_cond = self.model(x_in , t_in , c_in).chunk(2)
        e_t = e_t_uncond + uncond_scale * (e_t_uncond - e_t_cond)
        return e_t
    
    # this will be the sampling loop 
    # shape: is the shape of the generated model in the form  will contains the [batch_size , channels , width , height]
    # cond: conditional embeddings of
    # temp : temperature : random noise get multiplied
    # x_last : if not provided random noise will be lost
    # uncond_scale : unconditional guidance scale 
    # skip_steps : number of timestemps to skip
    # uncond_cond: conditional embeddings for the empty prompt c 
    # 
    def sample(self , shape: torch.Tensor , cond: torch.Tensor , temperature:float = 1 , skip_test:int = 1  , x_last:torch.Tensor = None , repeat_noise:bool = False):
        raise NotImplementedError('implements the parameter here too')  
        pass
    
    # this will be for the painting loop 
    # cond: conditional embeddings
    # t_start: sampling to start from T 
    # uncond_scale : the unconditional guidance scale 
    # uncond_cond : conditional embeddings for the c
    # skip steps: number of steps to skip
    def paint(self , x:torch.Tensor ,
               t_start:int ,
               uncond_scale:float =1  , 
               orign = None  ,
               uncond_cond: Optional[torch.Tensor] = None ,
               origin_noise:Optional[torch.Tensor] = None ,
               mask:Optional[torch.Tensor] = None
               ) :
        pass

    # sample from q 
    # x0: is x0(aplha) of shape
    # noise : is the noise ne
    # index: is the timestemps t
    
    def q_sample(self , x0:torch.Tensor , noise:torch.Tensor , index:None) : 
        raise NotImplementedError(' the q sample is not implemented check again')
        
        pass 
