import torch
import torch.nn as nn
from torch.nn import functional as F  
from encoder import VAE_ResidualBlock , VAE_AttentionBlock

class VaeDecoder(nn.Module) : 
    def __init__(self): 
        super().__init__(
            nn.Conv2d(4 , 4 , kernel_size=1 ,padding=0),
            nn.Conv2d(4 , 512 , kernel_size=3 , padding=1),
            VAE_ResidualBlock(512 , 512),
            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512 , 512),
            VAE_ResidualBlock(512 , 512),
            VAE_ResidualBlock(512 , 512),


            #repeat the rows and columns by scale factor N.B: resizing a image by doubling its size
            nn.Upsample(scale_factor=2),

            #batch size 512  height /4 width / 4 
            nn.Conv2d(512 , 512 , kernel_size=3 , padding=1),

            #batch size 512 height /2 width /2 
            VAE_ResidualBlock(512 , 512),
            #batch size 256 height /2 width /2
            VAE_ResidualBlock(256 , 256),
            VAE_ResidualBlock(256 , 256),
            
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256 , 256 , kernel_size=3 , padding=1),

            #batch size 256 , height , width 
            VAE_ResidualBlock(128 , 128),
            VAE_ResidualBlock(128 , 128),

            nn.GroupNorm(32 , 128),
            nn.SiLU(),
            nn.Conv2d(128 , 3, kernel_size=3 , padding=1)
        )

    def forward(self , x) : 
         x /= 0.18215

         for module in self : 
             x = module(x)

         return x
    


        