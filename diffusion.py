#  the code for the difusion model 
from torch import nn 
import torch 
from typing import List
from torch.nn import functional as F
from  encoder import VaeEncoder
from decoder import VaeDecoder
from Clipencoder import CLIP
from UnetArchitecture import Unet
from UnetArchitecture import DDPM
from Clipencoder import CLipTextEmbeddings
 
class DiffusionModelWrapper(nn.Module) : 
    def __init__(self , diffusion_model:Unet) : 
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self , x:torch.Tensor , time_steps:torch.Tensor ,context:torch.Tensor) : 
        return self.diffusion_model(x , time_steps , context)

#for the latent diffusion model 
#the unet model will means for the model architecture 
#the clip embedder will be used for the text embeddings 
# the latent scaler will be factor used for the latent space
# n_steps is the number of diffuison steps 
# linear start is the start of the Beta schedule
# linear end is the end of the Beta schedule 
class LatentDiffusionWrapper(nn.Module) : 
    def __init__(self , unetModel:Unet , autoencoder:VaeEncoder  , clip_embedder:CLipTextEmbeddings , latent_scaling_factor:float , n_steps:int , linear_start:float , linear_end:float) : 
        super().__init__()
        self.model = DiffusionModelWrapper(unetModel)
        self.first_stage = autoencoder
        self.latent_scaling_factor = latent_scaling_factor
        self.cond_stage = clip_embedder
        self.n_steps = n_steps 
        self.linear_start = linear_start
        self.linear_end = linear_end
        #self.noise = DDPM(Unet , 10)
        
#define the beta schedule
        beta = torch.linspace(linear_start ** 0.5 , linear_end ** 0.5 , n_steps , dtype=torch.float) ** 2 
        self.beta = nn.Parameter(beta.to(torch.float32) , requires_grad=False)
        alpha = 1 - beta 

        alpha_bar = torch.cumprod(alpha ,dim=0)
        self.aplha_bar = nn.Parameter(alpha_bar.to(torch.float32 ) , requires_grad=False)

 # get the clip embeddings for a list of text prompts
    def get_text_embedding(self , prompt:List[str]) : 
        return self.cond_stage(prompt)

 #get the latent space auto encoder of the image 
 #multiply the latentscalling factor by the autoencoder
    def  get_latent_autoencoder_encode(self , image:torch.Tensor) :
        return self.latent_scaling_factor * self.first_stage(image)
        

  #get the image from the latent space
    def get_latent_decoder(self, z:torch.Tensor) : 
        return self.second_stage(z / self.latent_scaling_factor)
    
  #predict noise  from the image
    def forward(self, x:torch.Tensor , t:torch.Tensor , context:torch.Tensor) : 
        return self.model(x , t ,context)

Okembsmodel = LatentDiffusionWrapper( unetModel=Unet  ,  clip_embedder=CLIP  , latent_scaling_factor=2 , n_steps=10 , linear_start=1e-2 , linear_end=2e-1)
model = DDPM(Okembsmodel)
print(model)
