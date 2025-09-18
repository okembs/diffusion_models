import torch 
from torch import nn 
from attentionVariant import SelfAttention , CrossAttention

class CLIPEmbedding(nn.Module) : 
    def __init__(self, n_vocab:int , n_embd:int , n_token:int):
        super().__init__()
        self.token_embeddings = nn.Embedding(n_vocab , n_embd)
        # a learnable weight matrix 
        self.position_embeddings = nn.Parameter(torch.zeros((n_token , n_embd)))

    def forward(self , tokens) : 
         x= self.token_embeddings(tokens)

         x += self.position_embeddings


# the clip player
class CLipPlayer(nn.Module) : 
    def __init__(self , n_head , n_embed) : 
        self.layernorm_1 = nn.LayerNorm(n_embed)
        #the self attention 
        self.attention = SelfAttention(n_embed)
        #pre fnn norm
        self.layernorm_2 = nn.LayerNorm(n_embed)
        #feed  forward layer 
        self.linear_1 = nn.Linear(n_embed , 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed , n_embed)

    def forward(self , x) : 
        residue = x 

        #batch size  , seq len 
        x = self.layernorm_1(x)
        x = self.layernorm_2(x)

        #batch size seq_len , dim
        x = self.attention(x)
        x += residue

        residue = x 
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        #Batch size seq_len 4 
        x = x * torch.sigmoid(1.702 * x )
        x = self.linear_2(x)
        x += residue
        return x 
    

class CLIP(nn.Module) : 
    def __init__(self):
        super().__init__()
        self.embeddings = CLIPEmbedding(49408 , 786 , 77)
        self.layers = nn.ModuleList([
            CLipPlayer(12 , 786) for i in range(12)
        ])
        self.layernorm = nn.LayerNorm(768)

    def forward(self , tokens): 
        tokens = tokens.type(torch.long)
        # Batch_size --> Batch size, seq Len , Dim
        state = self.embeddings(tokens)

        #apply the encoder layers similar  to the transformer encoders
        for layers in self.layers : 
            state = layers(state)
        output = self.layernorm(state)
        return output
