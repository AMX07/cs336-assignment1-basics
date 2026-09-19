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
    def __init__(self,d_model: int, eps: float = 1e-5 , device=None, dtype=None):
        '''
        '''
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
    
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self,theta:float,d_k: int, max_seq_len: int, device=None):
        super().__init__()
        """
        theta: float constant value
        d_k: int dim of query and key vectors
        max_seq_len: int max seqn length of the input
        device: torch.device device to store buffer on
        """
        self.angles = torch.tensor([[i/pow(theta,((2*k-2)/d_k)) for k in range(1,int(d_k/2)+1)] for i in range(max_seq_len)], device = device)
        # (max_seq_len, d_k/2)
        sines = torch.sin(self.angles)
        cosines =torch.cos(self.angles)
        self.register_buffer('sines', sines, persistent=False)
        self.register_buffer('cosines', cosines, persistent=False)
    def forward(self,x,token_positions):
        """
        x can have arbitary batch dims 
        tok positions are tensor (...,seq_len)
        use the token positions to slice your (possibly precomputed) cos and sin tensors along
        the sequence dimension.
        """
        rotated_even = (x[..., 0::2] * self.cosines[token_positions]) - (x[..., 1::2] * self.sines[token_positions])
        rotated_odd = (x[..., 0::2] * self.sines[token_positions]) + (x[..., 1::2] * self.cosines[token_positions])
        output = torch.empty_like(x)
        output[..., 0::2] = rotated_even 
        output[..., 1::2] = rotated_odd
        return output

def softmax(in_features,dim):
    values, indices = torch.max(in_features, dim=dim, keepdim=True)
    x = in_features - values
    return torch.exp(x)/torch.exp(x).sum((dim,),keepdim = True)

def scaled_dot_product_attention(q, k, v, mask=None):
        attention = ((q @ k.transpose(-2,-1))/sqrt(int(q.size(-1)))) 
        if mask is not None:
            attention = attention.masked_fill(~mask,float('-inf'))
        return softmax(attention,-1) @ v
           
class Multihead_self_attention(torch.nn.Module):
    '''
    d__model: int Dimensionality of the Transformer block inputs.
    num_heads: int Number of heads to use in multi-head self-attention.
    try combining the key, query, and value projections into a single weight matrix so you only need a
    single matrix multiply.
    need to apply RoPE
    '''
    def __init__(self,d_model, num_heads):
        super().__init__()
        self.W_Q = Linear(d_model,d_model) 
        self.W_K = Linear(d_model,d_model)
        self.W_V = Linear(d_model,d_model)
        self.W_O = Linear(d_model,d_model)

    def forward(self,x,num_heads,d_model, rope=None, token_positions=None):
        '''
        x,num_heads,d_model, rope=None, token_positions=None
        '''
        B,T,C = x.shape
        mask = torch.tril(torch.ones((T,T),dtype=torch.bool))
        head_size = d_model // num_heads

        if rope is not None:
            q = rope(self.W_Q(x).view(B,T,num_heads, head_size).transpose(1,2),token_positions)
            k = rope(self.W_K(x).view(B,T,num_heads, head_size).transpose(1,2),token_positions)
        else:
            q = self.W_Q(x).view(B,T,num_heads, head_size).transpose(1,2)
            k = self.W_K(x).view(B,T,num_heads, head_size).transpose(1,2)

        v = self.W_V(x).view(B,T,num_heads, head_size).transpose(1,2)
        return self.W_O((scaled_dot_product_attention(q,k,v,mask).transpose(1,2)).reshape(B,T,num_heads*head_size)) # B,num_heads,T,head_size  -> B,T,num_heads*head_size

class transformer_block(torch.nn.Module):
    def __init__(self,d_model,num_heads,d_ff):
        '''
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.

       2 sub-layers, 1: multihead self attention, 2 SwiGLU feed-forward network. 
       in every layer, RMSNorm -> (MHA/FF)-> residual connection.
        y = x + MultiHeadSelfAttention(RMSNorm(x))).
        '''
        super().__init__()
        self.rms1 = RMSNorm(d_model)
        self.rms2 = RMSNorm(d_model)
        self.mha = Multihead_self_attention(d_model,num_heads)
        self.ff = SwiGLU_FFN(d_model,d_ff)

    def forward(self,in_features,num_heads,d_model,rope):
        token_positions = torch.arange(start=0, end=in_features.size(-2), step=1, dtype=torch.long, device=in_features.device)
        x = self.mha(self.rms1(in_features),num_heads,d_model,rope,token_positions)
        x = x + in_features
        x = self.ff(self.rms2(x)) + x
        return x 
        

        

