import torch
import torch.nn as nn
import torch.functional as F
#the schedular for the unet arcitecture
# the denoisoing sampler  for the unet architecture 
#for denoising the image  or the unet architecture
# Denoising diffusion probabilistic models
#eps:epilson , n_steps is t
# this fucntion is used for denoising the data
class DDPM(nn.Module) : 
    def __init__(self, eps_model:nn.Module , n_steps: int , device:torch.device ):
          super().__init__()
          self.eps_model = eps_model
          self.n_steps = n_steps
          self.beta = torch.linspace(0.0001, 0.02 , n_steps).to(device)
          self.alpah = 1 - self.beta
          self.alpah_bar = torch.cumprod(self.alpah , dim=1)
          self.sigma = self.beta

   # get the q_x_to distribution
    def q_xt_0(self, x:torch.Tensor , t:torch.Tensor):  
       mean = gather(self.alpah_bar , t) ** 0.5 * x
       var = 1 - gather(self.aplha_bar , t)
       return mean , var
    #add noise to the data
    # the q_sample will be used for adding noise to the data
    # this will be used for add_noise  to the data
    def q_sample(self , x0:torch.Tensor , t:torch.Tensor , eps= None ) : 
       if eps is None : 
          eps = torch.rand_like(x0)
          mean , var = self.q_xt_0(x0 , t)
          return mean + (var ** 0.5) * eps
       #denoise the data
       # the  P_sample will be used for denoising the data
       # this will be used  for  denoising the data
    def p_sample(self , xt:torch.Tensor , t:torch.Tensor) : 
       eps_theta = self.eps_model(xt , t)
       alpha_bar = gather(self.alpah_bar ,t)
       alpha = gather(self.alpah , t)
       print(f'alpha here is : {alpha}')
       eps_coef = ( 1 - alpha) / ( 1 - alpha_bar) ** .5
       mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
       var = gather(self.sigma , t )
       eps = torch.randn(xt.shape , device=xt.device)
       #sample 
       return mean + (var ** .5) * eps 
    # for calculating the loss 
    def loss(self , x:torch.Tensor, noise = None) : 
        batch_size = x.shape[0]
        t = torch.randint(0 , self.n_steps, (batch_size) , device=x.device , dtype=torch.long)
        if noise is None : 
           noise = torch.randn_like(x)
        xt = self.q_sample(x , t , eps=noise)
        eps_theta = self.eps_model(xt , t)
        return F.mse_loss(noise , eps_theta)
      
   
# the gather function
def gather(const:torch.Tensor , t:torch.Tensor) : 
   c = const.gather(-1 , t)
   print(f"this is the gather value: {c}")
   return c.reshape(-1 , 1 , 1 , 1)
         
