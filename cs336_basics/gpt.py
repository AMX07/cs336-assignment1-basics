import torch
from math import sqrt
 
# basic building blocks
class Linear(torch.nn.Module):
    '''
    y = x @ W.t
    x # 1 x,d
    w.t # d,c
    d = in_features
    c = out_features
    '''
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__() 
        w_tensor = torch.empty(out_features,in_features,device=device,dtype=dtype)
        sd = sqrt(2/(in_features+out_features))
        torch.nn.init.trunc_normal_(w_tensor,0,sd,-3*sd,3*sd)
        self.weight = torch.nn.Parameter(w_tensor)

    def forward(self, x:torch.Tensor):
        y = x @ self.weight.T
        return y

class Embedding(torch.nn.Module):
    '''
    Creates an embedding look-up table 
    '''
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        emb_tensor = torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype)
        sd = 1
        torch.nn.init.trunc_normal_(emb_tensor,0,sd,-3*sd,3*sd)
        self.weight = torch.nn.Parameter(emb_tensor)
        #store the emb_table with d_model being the final dim

    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self,d_model: int, eps: float , device=None, dtype=None):
        super().__init__()
        g = torch.ones(d_model, device=device, dtype=dtype) # 64
        self.weight = torch.nn.Parameter(g)
        self.rms_a = lambda a: torch.sqrt(1/d_model * (torch.square(a).sum(-1, keepdim=True) )+ eps) #(4, 12,1 (because of keep dims))
        self.rms = lambda a, rmsa : (a*self.weight)/rmsa
           
    def forward(self,x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32) # for helping w/ overflowing or underflowing
        rmsa = self.rms_a(x) 
        x =  self.rms(x,rmsa)
        return x.to(in_dtype) 
     
class SwiGLU_FFN(torch.nn.Module):
    def __init__(self,d_model: int ,d_ff, device=None, dtype=None):
        super().__init__()
        # d_ff = 8/3 * d_model
        # d_ff = ((d_ff + 63) // 64) * 64
        self.w1 = Linear(d_model,d_ff, device, dtype)
        self.w2 = Linear(d_ff,d_model, device, dtype)
        self.w3 = Linear(d_model,d_ff, device, dtype)
        # self.weights = torch.nn.ParameterList(
        #     [torch.nn.Parameter(w.weight) for w in [self.w1,self.w2,self.w3]])
        self.SiLU = lambda x: x * torch.sigmoid(x)

        
    def forward(self, in_features: torch.Tensor):
        x = in_features
        x = self.w2(self.SiLU(self.w1(x)) * self.w3(x))
       
        return x 
    
# d_model,d_ff = 64, 64
# swiglu = SwiGLU_FFN(d_model,d_ff)
# weights = swiglu.w1.weight.data

# print(weights)