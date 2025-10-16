import torch 
from torch import nn 
from attentionVariant import SelfAttention , CrossAttention 
#converts the token text  to embedddings for it to generate the image
# we are supposed to use the tiktoken , but we want to use the openai tokenizer that will make the text embeddings to make it
from transformers import CLIPTokenizer , CLIPTextModel
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
    def __init__(self , device ,  max_text_length:80):
        self.device = device 
        self.max_text_length = max_text_length
        super().__init__()
   #    self.tokenizer = tiktoken.encoding_for_model('gpt-4o')


class CLipTextEmbeddings(nn.Module) : 
    def __init__(self ,  clip_model_version: str = 'openai/clip-vit-large-patch14' , device:str = "cuda:0" , max_length_str:int = 80) : 
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_version)
        self.device = device
        self.clip_str_version = clip_model_version
        self.max_length_str = max_length_str
        #define the clip transformer 
        self.transformer = CLIPTextModel.from_pretrained(self.clip_str_version)

    def forward(self, prompts:str) : 
        #tokenize the networks here 
        batch_encoding = self.tokenizer(prompts ,
                                         truncation=True ,
                                         max_length=self.max_length_str,return_length=True
                                        ,n_overflowing_tokens=False , padding="max length" , 
                                        return_tensors='pt')
        # get the tokenizer ids 
        tokens = batch_encoding['inputs_ids'].to(self.device)
        #return the clips embeddings 
        return  self.transformer(input_ids =tokens).last_hidden_state

