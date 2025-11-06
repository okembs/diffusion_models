import torch 
import torch.nn as nn 
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel , get_ema_avg_fn
from UnetArchitecture import Unet

#code the exponetial moving average to calculate the model 
# note the decay rate is beta and the value should be 0.999
def get_EMA_fn(decay=0.9999): 
    def ema(ema_param , current_param , num_average) : 
        return decay * ema_param + (1 - decay) * current_param
    return ema


# not the decay(beta) will be equal to  0.9999
def get_EMA_val(decay, diffusionModel) :
    return AveragedModel(diffusionModel , get_ema_avg_fn(decay) , use_buffers=True )
print(get_EMA_val(0.9 ,Unet))