#define the u-net architecture 
import torch 
from torch import nn
from torch.nn import functional as F
from labml_nn.diffusion.stable_diffusion.model.unet_attention import SpatialTransformer
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy
from attentionVariant import CrossAttention , SelfAttention
#from labml_nn.diffusion.ddpm.utils import gather

#the u net architecture contains the input_channels , out_channels , channels and n_resBlock
#the attention levels: are the levels the model is supposed to perform
#  n_heads : number of attention heads in the transformer 
#  , 
# tf_layers : the number of tf layers
# d_cond

class Unet(nn.Module) : 
    def __init__(self,  in_channels:int , out_channels:int , channels_multi:list[int]  ,channels:int ,  n_resBlock:int , attention_levels:list[int], n_heads:int, tf_layers:int = 1 , d_cond:int = 768 ):
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

     
     input_block_channels = [channels]
     channels_list = [channels * m for m in channels_multi]
    
     for i in range(levels) : 
        for _ in range(n_resBlock) : 
           #residual block maps from the previous block of channel
           layers = [ResBlock(channels , d_time_embd , out_channels=channels_list[i])]
           print(f'layers are : {layers}')
           channels = channels_list[i]
           # add the attention layers
           if i in attention_levels : 
              layers.append(SpatialTransformer(channels , n_heads , tf_layers , d_cond))
              

           self.input_blocks.append(TimeStepEmbdSequential())
           input_block_channels.append(channels)

           if i != levels - 1 : 
              self.input_blocks.append(TimeStepEmbdSequential(DownSample(channels)))
              input_block_channels.append(channels)
            
    # the midddle of the unet architecture
     self.middle_block = TimeStepEmbdSequential(
         ResBlock(channels , d_time_embd),
         SpatialTransformer(channels , n_heads , tf_layers , d_cond),
         ResBlock(channels , d_time_embd)

      )
     # for the second half of the unet
     
     self.output_blocks = nn.ModuleList([])
     for i in reversed(range(levels)) : 
        # add the residual attention
        for j in range(n_resBlock + 1 ) : 
           layers = [ResBlock(channels + input_block_channels.pop() , d_time_embd , out_channels=channels_list )]
           channels = channels_list[i]
           print(f'this is the layers in the neural net : {layers}')
           print(f'channels in the list is : {channels}')
           print(f'layers are  :{layers}')

           if i in attention_levels : 
              layers.append(SpatialTransformer(channels, n_heads, tf_layers , d_cond))

           if i != 0  and j == n_resBlock : 
              layers.append(Upsample(channels))
#add the output half of the unet    #and add the layers at the TimeStepEmbedSequential
     self.output_blocks.append(TimeStepEmbdSequential())

     self.out = nn.Sequential(
        normalization(channels),
        nn.SiLU(),
        nn.Conv2d(channels , out_channels , 3 , padding=1)
     )


    def  time_step_embeddings(self , time_steps:torch.Tensor , max_peroid:int = 1000) : 
        half = self.channels // 2 

        frequencies = torch.exp(
           -math.log(max_peroid) * torch.arange(start=0 , end=half , dtype=torch.float32)
           /half
        ).to(device=time_steps.device)

        args = time_steps[: , None].float() * frequencies[None]
        return torch.cat([torch.cos(args) , torch.sin(args)] , dim=1)
    

    def forward(self , x:torch.Tensor , time_steps:torch.Tensor , cond:torch.Tensor) : 
       x_input_Block = []
       t_emd =self.time_step_embeddings(time_steps)
       t_emd = self.time_embd(t_emd)
       for module in self.input_blocks : 
          x = module(x , t_emd, cond)
          x_input_Block.append(x)

       x = self.middle_block(x , t_emd, cond)

       for module in self.output_blocks : 
          x = torch.cat([x , x_input_Block.pop()] , dim=1)
          x = module(x , t_emd, cond)

       return self.out(x)
     

# define the timesetpeEmdedSequential 
#note this module can contain  different module like the attention block , resnlock and so on
class TimeStepEmbdSequential(nn.Sequential):      
   #take the different layers here
   def forward(self, x ,t_emb , cond=None): 
      for layer in self : 
         if isinstance(layer , ResBlock) : 
            x = layer(x , t_emb)
         elif isinstance(layer , SpatialTransformer) : 
            x = layer(x , cond)
         else :
                x = layer(x)

      return x 
   


# this makeup a resnet block
#for the Resnet block on how to fix it
# channels : is  the number of input channels 
# d_temb: is the  number of the timestep embeddings
# out_channels : is the number of the outchannels
class ResBlock(nn.Module) : 
   def __init__(self, channels , d_temd , out_channels: None | int =None ):
     super().__init__()


     if out_channels is None : 
        out_channels = channels
        print(f'out channels is : {out_channels}')

#the first normalization and convolution
     self.in_layers = nn.Sequential(
        normalization(channels),
        nn.SiLU(),
        nn.Conv2d(channels , out_channels, 3, padding=1)
      )
  
# for the timeStempEmbeddings
     self.emb_layers = nn.Sequential(
        nn.SiLU(),
        nn.Linear(d_temd , out_channels)
     )
#for the final convolution layer
     self.out_layers = nn.Sequential(
        normalization(out_channels),
        nn.SiLU(),
        nn.Dropout(0.5),
        nn.Conv2d(out_channels , out_channels , 3 ,padding=1)
     )

     if out_channels == channels : 
        self.skip_connection = nn.Identity()
     else :
         self.skip_connection = nn.Conv2d(channels , out_channels , 1)

   def forward(self , x:torch.Tensor , t_emb:torch.Tensor) : 
      h = self.in_layers(x)
      t_emb = self.emb_layers(t_emb).type(h.type) # add the timestemp embeddings
      h = h + t_emb[: , : ,None , None]
      h = self.out_layers(h) # for the final convolution
      return self.skip_connection(x) + h 





#for the downsample
class DownSample(nn.Module) : 
   def __init__(self, channels:int) :
      super().__init__()
      self.op = nn.Conv2d(channels , channels , 3 , stride=2 , padding=1)

   def forward(self , x:torch.Tensor) : 
      return self.op(x)

      

#for the upSample
class Upsample(nn.Module) : 
   def __init__(self , channels:int) :
          super().__init__()
          self.conv = nn.Conv2d(channels , channels , 3, padding=1)
   def forward(self , x:torch.Tensor) : 
      x = F.interpolate(x , scale_factor=2 , mode='nearest')
      return self.conv(x)
   
# the group normalisation with the 32 casting 
class GroupNorm32(nn.GroupNorm) : 
   def forward(self , x) : 
      return super().forward(x.float()).type(x.dtype)

# a function for the group normalization
def normalization(channels) : 
   return GroupNorm32(32 , channels)
 
#test sinusodial for the timestepembeddings 
def test_time_embeddings() : 
   plt.figure(figsize=(15 ,5))
   m = Unet(in_channels=1 , out_channels=1 , channels=320 , n_resBlock=1 , attention_levels=[] , tf_layers=1 , d_cond=1 , channels_multi= [] , n_heads=2)
   te = m.time_step_embeddings(torch.arange(0, 1000))
   plt.plot(np.arange(1000) , te[: , [50 , 100 , 190 , 260]].numpy())
   #plt.legend(['dim %d' % for p in [50 , 100 , 190 , 260]])
   plt.title('time embeddings')
   plt.show()


print(test_time_embeddings())
\


   
    
       