import torch 
import torch.nn as nn 
from torch.nn import functional as F




def scale_dot_product_attention(Q , K , V , mask=None) : 
    d_k = Q.size(-1)
    scores = torch.matmul(Q , K.transpose(-2 , -1)) // torch.sqrt(torch.tensor(d_k , dtype=torch.float32))
    if mask is not None : 
        scores = scores.masked_fill(mask == 0 , float('-inf'))

        #softmax to normalizes scores  , producing attention weight
        attention_weight = F.softmax(scores, dim=1)

        #compute final values as weighted values 
        output = torch.matmul(attention_weight , V)
        return attention_weight , output

class SelfAttention(nn.Module) : 
    def __init__(self, embed_size):
        super(SelfAttention).__init__()
        self.embed_size = embed_size

        #define the linear qualities Q(query) , V(value) and k(key)
        self.query = nn.Linear(embed_size , embed_size)
        self.key = nn.Linear(embed_size , embed_size)
        self.value = nn.Linear(embed_size , embed_size)

    def forward(self , x) : 
        # Generate the Q , K , V matrices
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        #calculate the attention using our scaled dot-product
        out, _ = scale_dot_product_attention(Q , K , V)
        return out






# the cross attention implementation in pytorch
# d_model : input embedding size
#n_heads: number of attention heads
#d_cond : size of attention head
#is_inplace: specify  whether to perform the attention softmax or save memory

#and we will still define Q(query) , k(keys) and v(values) here
class CrossAttention(nn.Module) : 
    use_flash_attention:bool = False
    def __init__(self, d_model: int , n_heads :int ,  d_cond:int ,d_heads:int   ,is_inplace:bool = True):
        super().__()
        self.d_model = d_model 
        self.n_head = n_heads 
        self.d_heads = d_heads
        self.d_cond  = d_cond 
        self.scale = d_heads * 0.5
        self.is_inplace = is_inplace

        d_attn = d_heads * n_heads
        self.to_q = nn.Linear(d_model, d_attn , bias=False)
        self.to_k = nn.Linear(d_cond , d_attn , bias=False)
        self.to_v = nn.Linear(d_cond , d_attn , bias=False)

        self.to_out = nn.Sequential(nn.Linear(d_attn , d_model))
        #set the flash attention to None  
        self.flash = None 

    def forward(self ,x:torch.Tensor , cond) : 
        has_cond = cond is not True
        if not has_cond : 
            cond = x 
        
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        #since i'm not using the flass attention use the normal attention 
        self.normal_attention(q , k , v)
    #for the flash attention
    # q are the query vectors before splitting heads of shape 
    #k are the query vectors before splitting heads of shape 
    #v are the query vectors before splitting heads of shape 

    def flash_attention(self , q:torch.Tensor , k:torch.Tensor , v:torch.Tensor) : 


        batch_size  , seq_len , _ = q.shape

        qkv = torch.stack((q , k , v ) , dim=2)

        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_heads)

        if self.d_heads <= 32 : 
            pad = 32 - self.d_heads 
        elif self.d_heads <= 64 : 
            pad = 64 - self.d_heads 
        elif self.d_heads < 128 : 
            pad = 128 - self.d_heads
        else : 
            raise ValueError(f"head size : {self.d_heads} too large for flash attention")
        
        if pad : 
            qkv = torch.cat((qkv , qkv.new_zeros(batch_size , seq_len , 3 , self.n_head, pad)) , dim=1)


         # call the flash module here
        out , _ = self.flash(qkv)

        out = out.reshape(batch_size,seq_len , self.n_head * self.d_heads)
        return self.to_out(out)
    
    # for the normal function 
    def normal_attention(self , q:torch.Tensor , k:torch.Tensor , v:torch.Tensor) : 
        q = q.view(*q.shape[:2], self.n_head , -1)
        K = k.view(*k.shape[:2] , self.n_head , -1)
        v = v.view(*v.shape[:2] , self.n_head , -1)

        attn = torch.einsum('bihd,bjhd->bhij', q , k) * self.scale

 # compute the softmax
        if self.is_inplace : 
            half = attn.shape[0] // 2 
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[half:].softmax(dim= -1)
        else : 
            attn = attn.softmax(dim= -1)

        out = torch.einsum('bihd,bjhd->bhij' , attn , v)
        #compute attention output
        out = out.reshape(*out.shape[:2] , -1)
        return self.to_out(out)


#the  feedforward network 
# d_model: is the input embedding size 
#d_multi : is a multiplicative factor for the hidden layer size
#  we will create an activation for the GeLu 
class FeedForward(nn.Module) : 
    def __init__(self, d_model:int , d_multi:int = 4):
        super().__init__()
        self.net = nn.Sequential(
            GeLU(d_model , d_model * d_multi),
            nn.Dropout(0.),
            nn.Linear(d_model * d_multi , d_multi)
        )

    def forward(self , x:torch.Tensor) : 
        return self.net(x)


# define the GeLU activation number
class GeLU(nn.Module) : 
    def __init__(self, d_in:int , d_out:int):
        super().__init__()
        self.d_in = d_in 
        self.d_out = d_out 
        
        #define the linear projections
        self.proj = nn.Linear(d_in , d_out * 2)

    def forward(self, x:torch.Tensor) : 
        x , gate  = self.proj(x).chunk(2 , divmod=1)
        return x * F.gelu(gate)