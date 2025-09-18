from torch import nn 
import torch 
from torch.nn import functional as F
from attentionVariant import SelfAttention

#this will contain the vae encoder 

class VaeEncoder(nn.Module) : 
    def __init__(self)  : 
        super.__init__(
            #batch size for the channel , width and height
            nn.Conv2d(3 , 128 , kernel_size=3 , padding=1),
            # batch size for the residual block 
            VAE_ResidualBlock(128 , 128),
            VAE_ResidualBlock(128 , 128),

            #batch size 128  width/2 same as height/2
            nn.Conv2d(128 , 128 , kernel_size=3 , padding=0),
            #batch size for the residual block 128 , 128
            VAE_ResidualBlock(128 , 256),
            VAE_ResidualBlock(256 , 256),

            #batch size 256 width/2 height/2
            nn.Conv2d(256 , 256 , kernel_size=3 , padding=0),
            #batch size 256  width/4 height/4
            VAE_ResidualBlock(256 , 152),
            #batch size 256
            VAE_ResidualBlock(256 , 256),

            #batch size 512 height/8 width/8
            VAE_ResidualBlock(512 , 512),
            VAE_ResidualBlock(512 , 512),
            VAE_ResidualBlock(512 , 512),

            #the  attention Block
            VAE_AttentionBlock(512 , 512),
            VAE_ResidualBlock(512 , 512),
            nn.GroupNorm(32 , 512),

            nn.SiLU(),

            #the batch size 512  and height /8 with width /8 
            nn.Conv2d(512 , 8 , kernel_size=3  , padding=1 ),
            #batch size 8 width/8 height/8
            nn.Conv2d(8 , 8 , kernel_size=3 , padding=1)
        )


    def forward(self , x :torch.tensor,  noise) : 
            for module in self : 
                if getattr(module , 'stride' , None) == (2 ,2):
                    x = F.pad(x , (0 ,1 , 1 , 0)) 
                x = module() 
                
                #batch size
                mean , log_variance = torch.chunk(x , 2 , dim=1)
                log_variance = torch.clamp(log_variance , -30 , 20)
                variance = log_variance.exp()
                stdev = variance.sqrt()

                # Z = N( 0 , 1) -->X =  N(mean , variance)
                x = mean + stdev * noise
                x *= 0.8125 
                return x





# the vae residual block
class VAE_ResidualBlock(nn.Module):
    def __init__(self , in_channels , out_channels): 
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32 , in_channels)
        self.conv_1 = nn.Conv2d(in_channels , out_channels , kernel_size=3 , padding=1)

        self.groupnorm_2 = nn.GroupNorm(32 , in_channels)
        self.conv_2 = nn.Conv2d(in_channels , out_channels)

        if in_channels == out_channels : 
            self.residual_layer = nn.Identity()
        else : 
            self.residual_layer = nn.Conv2d(in_channels , out_channels , kernel_size=1 , padding=0)


    def forward(self , x) : 
        residue  = x 
        x = self.groupnorm_1(x)
        x = F.silu(x)

        # Batch size  , in_channels, height and width 
        x = self.conv_1(x)

        #batch size 
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)
    


# the attention Block
class VAE_AttentionBlock(nn.Module) : 
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32 , channels)
        self.attention = SelfAttention(1 , channels)

    def forward(self ,x) : 
        residue = x 
        x = self.groupnorm(x)
        n , c, h , w = x.shape

        #Batch size features and height 
        x = x.view((n , c , h * w))
        #Batch size features height and width
        x = x.transpose(-1 , -2)

        x =  self.attention(x)

        x = x.transpose(-1 , -2)
        #Batch size 
        x = x.view((n, c , h ,w))
        #Batch size features height and width 
        x += residue

        return x 
    
    