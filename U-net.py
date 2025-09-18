#define the u-net architecture 
import torch 
from torch import nn
from torch.nn import functional as f
from labml_nn.diffusion.stable_diffusion.model.unet_attention import SpatialTransformer

#the u net architecture contains the input_channels , out_channels , channels and n_resBlock
#the attention levels n_heads  , tf_layers d_cond
class Unet(nn.Module) : 
    def __init__(self,  in_channels:int , out_channels:int , channels_multi:list[int]  ,channels:int ,  n_resBlock:int , attention_levels:list[int], n_heads:list[int], tf_layers:int = 1 , d_cond:int = 768 ):
     super().__init__()
     self.channels  = channels
     self.in_channels = in_channels
     self.out_channels = out_channels
     self.channels_multi = channels_multi
     self.n_resBlock = n_resBlock
     self.attention_levels = attention_levels
     self.n_heads = n_heads
     self.tf_layers = tf_layers
     self.d_cond = d_cond

     #define all parameters like the levels 
     levels = len(channels_multi)
     d_time_embd = channels * 4 
     self.time_embd = nn.Sequential(
        nn.Linear(channels , d_time_embd),
        nn.SiLU(),
        nn.Linear(d_time_embd , d_time_embd)
     )

     self.input_blocks = nn.ModuleList()
     self.input_blocks.append(TimeStepEmbdSequential(
        nn.Conv2d(in_channels , out_channels , 3 , padding=1)
     ))

     input_block = [channels]
     channels_list = [channels * m for m in channels_multi]
     for i in range(levels) : 
        for _ in range(n_resBlock) : 
           layers = [ResBlock(channels , d_time_embd , out_channels=channels_list[i])]
           channels = channels_list[i]
           if i in attention_levels : 
              layers.append(SpatialTransformer(channels , n_heads , tf_layers , d_cond))
              self.input_blocks.append(TimeStepEmbdSequential(*layers))
              


# define the timesetpeEmdedSequential 
class TimeStepEmbdSequential(nn.Module): 
   pass


class ResBlock(nn.Module) : 
   pass