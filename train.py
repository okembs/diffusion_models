# for training the diffusion model
import torch 
import torch.nn as nn 
import torch.functional as F 
from torch.optim import AdamW , Adam
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from Dppmclassifier import DDPM
from UnetArchitecture import Unet
from calculations import get_EMA_val
class Customdatasets : 
    def __init__(self , corn ) : 
        self.corn = corn
        pass 

Parameters = {
    "in_channels": 1,
     "output_channels": 1 ,
     "numgroups": 32 ,
     'num_heads': 8 ,
     "time_steps": 1000,
     "AttentionLevels": [False , True , False , True , False ],
     "channelsMultipliers": [320 , 640 , 1280 , 1280] ,
     "n_resblock": 2,
     "tf_layers": 2,
      "d_cond": 768
}

# define the hyper parameters when training it 
learning_rate = 1e-3
decay = 0.9999



# for training the model 
def train(batch_size:int , 
          num_steps:int , 
          num_epochs:int , 
          seed:int , 
          ema_decay:float ,
          checkPoint_path:str
          ) :
    
    train_datasets = None 
    test_datasets = None
    scheduler = DDPM.cosine_beta_schedular(num_steps)
    model = Unet(in_channels=Parameters["in_channels"] , out_channels=Parameters['output_channels'] , channels_multi=Parameters['channelsMultipliers'] , n_heads=Parameters['num_heads'] , tf_layers=Parameters['tf_layers'])
    optimiser = AdamW(model.parameters() , lr=learning_rate) 
    # for the exponetial moving avarage to stabilize model training and improve generalization 
    EMA =  get_EMA_val(decay=decay , diffusionModel=model)
    