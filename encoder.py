from torch import nn 
import torch 
from torch.nn import functional as F
from attentionVariant import SelfAttention , CrossAttention
from UnetArchitecture import ResBlock , normalization
from typing import List
import dataclasses
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


    def forward(self , x :torch.tensor, noise) : 
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
    



#let this be for the autoencoder module 
#z  : is the number of latent space
# emb_channels : is the number of the embeddings in the latent space
#  , z_channels
class Autoencoder(nn.Module) : 
    def __init__(self, encoder: 'Encoder', decoder: 'Decoder' , emb_channels:int , z_channels:int):
        super().__init__()
        self.encoder = encoder 
        self.decoder = decoder 
        self.emb_channels = emb_channels
        self.z_channels = z_channels

# convolution to map the quantum space to embeddings
        self.quant_conv = nn.Conv2d(2 * z_channels , 2 * emb_channels , 1)
# post convolution after the quantum spaced is mapped 
        self.post_quant_conv = nn.Conv2d(emb_channels , z_channels , 1)

# for the encode
    def encode(self , img:torch.Tensor) : 
        z = self.encoder(img)
        moments = self.quant_conv(z)
        return GaussanDistribution(moments)
#for the decode : decode images from latent representation
# z is the latent space with embeddings with batch_size , embsize
    def  decode(self , z:torch.Tensor) : 
        z = self.post_quant_conv(z)
        #decode the image shape
        return self.decoder(z)
    
#the encoder contains the channels , 
# channels_multipliers , 
# n_res_Block : number of resnet block , 
# in_channels: number of input channels ,
#  z_channels : number of the channels in the embedded space
class Encoder(nn.Module) : 
    def __init__(self, n_res_Block:int , in_channels:int , z_channels:int , channels_multipliers: List[int] , channels):
        super().__init__()
        self.n_res_Block = n_res_Block
        self.channels = channels
        self.in_channels = in_channels
        self.z_channels = z_channels 
        self.channels_multipliers = channels_multipliers

        n_resolution = len(channels_multipliers)
        #a convolution layer that maps the image to channel 3 * 3 
        self.conv_in = nn.Conv2d(in_channels , channels , 3 , padding=1 , stride=1)
        channel_list = [m * channels for m in [1] + channels_multipliers]
        self.down = nn.ModuleList( )
        for i in range(n_resolution) : 
            resnet_blocks = nn.ModuleList()
            for _ in range(resnet_blocks) : 
                resnet_blocks.append(ResnetBlock(channels , channel_list[i + 1]))
                channels = channel_list[i + 1]
        down = nn.Module()
        down.block = resnet_blocks
        if i != n_resolution - 1 : 
            down.downsample = DownSample(channels)
        else : 
            down.downsample = nn.Identity()
        self.down.append(down)

        #final resnet block with attention
        self.mid = nn.Module()
        self.mid_block1 = ResBlock(channels= channels , out_channels=channels , d_temd=None )    # ResnetBlock(channels , channels)
        self.mid_attn = CrossAttention(320 , 5 , 64 , 64)
        self.mid_block2 = ResBlock(channels=channels ,out_channels=channels , d_temd=None)
        #map to embeddings space with a 3 x 3 convolution
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels , 2 * z_channels , 3 , stride=1, padding=1)

        #img : image tensor with shape
    def forward(self , img:torch.Tensor) : 
        x =  self.conv_in(img)
        for down in self.down : 
            for block in down.block : 
                x = block(x)
            x = down.downsample(x)
            #final resnetblock with attention 
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        #normalize and map to the embeddings space 
        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)
        return x 
    



class Decoder(nn.Module) : 
    def __init__(self,  n_res_Block:int , in_channels:int , out_channels:int  ,z_channels:int , channels_multipliers: List[int] , channels):
        super().__init__()
        self.n_res_Block = n_res_Block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.channels_multipliers = channels_multipliers
        self.channels = channels
        #define the rest here
        n_resolution = len(channels_multipliers)
        channels_list = [m * channels for m in channels_multipliers]
        print(channels_list)
        channels = channels_list[-1]
        self.conv_in = nn.Conv2d(z_channels , channels , 3 , padding=1 , stride=1)
        #the resnet block with attention 
        self.mid = nn.Module()
        self.mid_block1 = ResBlock(channels=channels ,out_channels=channels , d_temd=None)
        self.mid_attn1 = CrossAttention(320 , 5 , 64, 64)
        self.mid_block2 = ResBlock(channels=channels , out_channels=channels , d_temd=None)
        #List of top levels block 
        self.up = nn.ModuleList()
        for i in reversed(range(n_resolution)) : 
            resnet_block =  nn.ModuleList()
            
            for _ in range(n_res_Block + 1) : 
                resnet_block.append(ResBlock(channels=channels , out_channels=channels_list[i] , d_temd=None))
                channels = channels_list[i]

            up = nn.Module()
            up.block = resnet_block
            # uapsmapling each of the block except the first
            if i != 0 : 
                up.Upsample = Upsample(channels)
            else : 
                up.Upsample = nn.Identity()
                
            self.up.insert(0 , up)

            #map to image with a 3 x 3 convolution
            self.norm_out = normalization(channels)
            self.conv_out = nn.Conv2d(channels , out_channels , stride=1 , padding=3)
    def forward(self, x:torch.Tensor) : 
        h = self.conv_in(x)
        h = self.mid_block1(x)
        h =  self.mid_attn1(x)
        h = self.mid_block2(x)
        #Top level blocks
        for up in reversed(self.up) : 
            # for the Resnet Block
            for block in up.block : 
                h = block(h)        
            h = Upsample(h)
     #normalize and map image space 
        h = self.norm_out(h)
        h = swish(h)
        img = self.conv_out(img)
        return img

# the gaussan distributor
class GaussanDistribution(nn.Module) : 
    pass


class ResnetBlock : 
    def __init__(self , in_channels, d_temb , out_channels:None = int):
        #the first layers and normalization 
        self.in_layers = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels , out_channels , 3 , padding=1)
        )
        pass 

class DownSample(nn.Module) : 
    def __init__(self, in_channels:int) :
        self.net = nn.Conv2d(in_channels , in_channels , 3, stride=2 , padding=1)
    def forward(self , x:torch.Tensor) : 
        x = self.net(x)
        return x

class Upsample(nn.Module) :
    def __init__(self , channels:int) : 
        self.conv = nn.Conv2d(channels , channels , 3 , padding=1)
    def forward(self , x:torch.Tensor) : 
        x = F.interpolate(x , scale_factor=2 , mode='nearest')
        return  self.conv(x)

#for the swish activation 
# we use the swish activation because it better for optimization than relu
def swish(x:torch.Tensor) : 
    return x *  torch.sigmoid(x)

