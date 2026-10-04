import torch
from math import sqrt
import math

# basic building blocks
class Linear(torch.nn.Module):
    '''
    x (B,T,C) : x can have any number of dims

    W(out_features, in_features) always 2D
    
    (C must be = in_features)
    
    x = x @ W.T  # B,T,C @ in_features, out_features = B,T,out_features

    weights initialized with: 𝒩︀(𝜇 = 0, 𝜎2 = 2/𝑑in+𝑑out) truncated at [−3𝜎, 3𝜎].
    '''
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__() 
        w_tensor = torch.empty(out_features,in_features,device=device,dtype=dtype)
        sd = sqrt(2/(in_features+out_features))
        torch.nn.init.trunc_normal_(w_tensor,0,sd,-3*sd,3*sd)
        self.weight = torch.nn.Parameter(w_tensor)

    def forward(self, x:torch.Tensor):
        return x @ self.weight.T
        
class Embedding(torch.nn.Module):

    '''
    vocab_size,embedding_dim= d_model
    creates an embedding look-up table
    weights initialized with : 𝒩︀(𝜇 = 0, 𝜎2 = 1) truncated at [−3𝜎, 3𝜎].
    '''
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        emb_tensor = torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype)
        sd = 1
        torch.nn.init.trunc_normal_(emb_tensor,0,sd,-3*sd,3*sd)
        self.weight = torch.nn.Parameter(emb_tensor)

    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self,d_model: int, eps: float = 1e-5 , device=None, dtype=None):
        '''
        input = output = (batch_size, sequence_length, d_model)
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
    '''
    d_ff = 8/3 * d_model = 4 * 2/3
    2/3 for scaling down for GLU variants 
    
    '''
    def __init__(self,d_model: int ,d_ff, device=None, dtype=None):
        '''
        Simple FFN:
        x → Linear → ReLU → Linear → output

        SwiGLU FFN:
        x → W₁ → SiLU ──┐
                        × → W₂ → output
        x → W₃ ─────────┘
        input = output ...,d_model
        '''
        super().__init__()
        self.w1 = Linear(d_model,d_ff, device, dtype)
        self.w2 = Linear(d_ff,d_model, device, dtype)
        self.w3 = Linear(d_model,d_ff, device, dtype) # gating mechanism
        self.SiLU = lambda x: x * torch.sigmoid(x)        
        
    def forward(self, in_features: torch.Tensor):
        # serial computation, can be parallelized.
        return self.w2(self.SiLU(self.w1(in_features)) * self.w3(in_features))
    
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self,d_k: int, max_seq_len: int,theta : float = 10000.0, device=None):
        super().__init__()
        """
        theta: float constant value
        d_k: int dim of query and key vectors
        max_seq_len: int max seqn length of the input
        device: torch.device device to store buffer on
        """
        self.angles = torch.tensor([[i/pow(theta,((2*k-2)/d_k)) for k in range(1,int(d_k/2)+1)] for i in range(max_seq_len)], device = device)
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
        input and output  = (..., seq_len, d_k)
        """
        rotated_even = (x[..., 0::2] * self.cosines[token_positions]) - (x[..., 1::2] * self.sines[token_positions])
        rotated_odd = (x[..., 0::2] * self.sines[token_positions]) + (x[..., 1::2] * self.cosines[token_positions])
        output = torch.empty_like(x)
        output[..., 0::2] = rotated_even 
        output[..., 1::2] = rotated_odd
        return output

def softmax(in_features,dim):
    '''
   softmax(x)i = exp(xi)/(∑n,j=1 exp(xj))

   Args:
           in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
           dim (int): Dimension of the `in_features` to apply softmax to.
   
       Returns:
           Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
           softmax normalizing the specified `dim`.
    '''
    values, indices = torch.max(in_features, dim=dim, keepdim=True,out=None)
    
    x = in_features - values
    counts = torch.exp(x)
    probs = counts / counts.sum((dim,),keepdim=True)
    # x = /torch.exp(x).sum((dim,),keepdim = True)
    return probs

def cross_entropy(inputs, targets):

    # subtracting the max values
    shifted = inputs - inputs.max(dim = -1)
    log_sum_exp = shifted.exp().sum(dim=-1).log()
    target_logits = shifted.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)

    return (log_sum_exp - target_logits).mean()
    
def scaled_dot_product_attention(q, k, v, mask=None):
        attention = ((q @ k.transpose(-2,-1))/sqrt(int(q.size(-1)))) # (B,num_heads,T, head_size) @ (B,num_heads,head_size,T) = (B,num_heads,T,T),, @ v (B,num_heads,T, head_size) = (B,num_heads,T, head_size)
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
        # remove num_heads
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
        return self.W_O((scaled_dot_product_attention(q,k,v,mask).transpose(1,2)).reshape(B,T,num_heads*head_size)) # B,num_heads,T,head_size  -> B,T,num_heads*head_size = B,T,d

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

class transformer_lm(torch.nn.Module):

    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta):
        '''
        vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the
        token embedding matrix.
        context_length: int The maximum context length, necessary for determining the dimensionality
        of the RoPE sin and cos buffer.
        num_layers: int The number of Transformer blocks to use.

        vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            weights: dict[str, Tensor],
            in_indices: 
        
        '''
        super().__init__()

        self.tok_embd = Embedding(vocab_size,embedding_dim= d_model)
        
        self.rope = RotaryPositionalEmbedding(rope_theta,d_model // num_heads, context_length)

        self.blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(transformer_block(d_model,num_heads,d_ff))
        self.norm = RMSNorm(d_model)
        self.output_layer = Linear(d_model,vocab_size)  #lm head layer 
 

    def forward(self,in_indices,num_heads,d_model):
        
        x = self.tok_embd(in_indices)  #B,seq_len -> B, seq_len, d_model
        for block in self.blocks:
            x = (block(x,num_heads,d_model,self.rope))
        x = self.output_layer(self.norm(x))
        # shouldnt we be using the embedding table for the output layer?
        return x
    
    def flops(vocab_size,context_length,num_layers,d_model,num_heads,d_ff,rope_theta,B):
        T =context_length
        head_size = d_model // num_heads

        # Per sequence, per transformer block
        qkv_flops = 3 * (2 * T * d_model**2)
        attention_flops = num_heads * (2 * 2 * T**2 * head_size)
        output_projection_flops = 2 * T * d_model**2

        mha_flops = qkv_flops + attention_flops + output_projection_flops
        swiglu_ffn_flops = 3 * (2 * T * d_model * d_ff)

        transformer_block_flops = mha_flops + swiglu_ffn_flops
        lm_head_flops = 2 * T * d_model * vocab_size

        # Total dense matrix-multiplication FLOPs for the batch
        B * (num_layers * transformer_block_flops + lm_head_flops) # 3516769894400

'''
Total_Flops_forward = Batch_size * (num_layes * transformer_block_flops + LM_head_flops)

LM_head_flops = output_layer_flops = 2*seq_len*

transformer_block_flops:
1.MHA: q,k,v projections + attention computation + output_projection
    = (3 * (2(Seq_len)(d_m)^2) +( 2* 2(seq_len)^2*(head_size) )+ 2(seq_len)(d_m)^2

2.swiglu_ffn: 2* 2(seq_len)(d_m)(d_ff) + 2(seq_len)(d_ff)(d_m)

'''

model = transformer_lm(
    vocab_size=50257,
    context_length=1024,
    num_layers=48,
    d_model=1600,
    num_heads=25,
    d_ff=4288,
    rope_theta=10000.0,
)

total_trainable_params = sum( p.numel() for p in model.parameters() if p.requires_grad)
params =  sum(p.numel() for p in model.parameters()) 

# print(total_trainable_params) #1640452800
# so 1640452800 f32 floating points take-up 1640452800 * 4 bytes
# which approx 6.5 GB

'''
total_flops_forward = B * (
    num_layers * transformer_block_flops + lm_head_flops
)
num_layers = 12
d_model = 768
num_heads = 12
print(f"gpt2-small: {total_flops_forward}") # gpt2-small: 1840726016000


total_flops_forward = B * (
    num_layers * transformer_block_flops + lm_head_flops
)
num_layers = 24
d_model = 1024
num_heads = 16
total_flops_forward
print(f"gpt2-mid: {total_flops_forward}") # gpt2-mid: 1002704076800

total_flops_forward = B * (
    num_layers * transformer_block_flops + lm_head_flops
)
num_layers = 36
d_model = 1280
num_heads = 20
total_flops_forward
print(f"gpt2-large: {total_flops_forward}") # gpt2-large: 1840726016000


(e)

16384/1024 # 16x more content lenght
133577729638400/3516769894400 # = 38
'''


#Optimizers
from collections.abc import Callable, Iterable
from typing import Optional

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        '''
        Slight variation of SGD where the lr decays over training, so we take succesively smaller steps over time.
        params: learnable params, like weights in linear layers, they might come in groups, each with different hyperparams. 
                if they come as single collections, the base contructor will assign them a default hyperparam like:
        lr = 1e-3
        '''
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        '''
        we iterate over each param in each group to apply the SGD: θ_{t+1} = θ_t - (α / √(t + 1)) ∇L(θ_t; B_t)

        iteration number is kept as a state.

        The torch.optim.Optimizer API specifies that the user might pass in a callable closure to re-compute the loss before the
        optimizer step. We dont use it by we are passing it so as to comply with the API
        '''
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss

class adamw(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        '''
        AdamW adds 2 moment vectors m and v as additional optimizers states, which allows for more sophisticared optimization.
        hyperparameter like betas 1 and 2 are used for calulating moment estimates.
        
        AdamW improves Adam regularization by adding weight decay (at each iteration, we pull the parameters
        towards 0), in a way that is decoupled from the gradient update.
        '''
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas" : betas, "eps" : eps, "weight_decay" : weight_decay}
        super().__init__(params, defaults) # add the above here with same shapes #m,v, theta, all same shape

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]


            for p in group["params"]:
                if p.grad is None:
                    continue
                #Sample batch of data 𝐵𝑡, update          
                state = self.state[p] # Get state associated with p. 

                t = state.get("t", 1) # Get iteration number from the state, or 1.
                if "m" in state:
                    m = state["m"]
                    v = state["v"]
                else:
                    m = torch.zeros_like(p)
                    v = torch.zeros_like(p)
                    

                grad = p.grad.data # Get the gradient of loss with respect to p.

                lr_t = lr * (math.sqrt(1- pow(betas[1],t)))/ (1-pow(betas[0],t)) 

                p.data -= lr * weight_decay * p.data # apply weight decay rate  # 2N flops

                m = betas[0]*m + ((1-betas[0]) * grad) # 3 N flops

                v = betas[1]*v + ((1-betas[1]) * pow(grad,2))

                p.data -= lr_t * m/(v.sqrt() + eps)

                state["m"] = m   
                state["v"] = v           
                state["t"] = t + 1 # Increment iteration number.

        return loss

# training loop
weights = model.parameters()
# opt = SGD([weights], lr=1e3) #1e1

# for t in range(10):
#     opt.zero_grad() # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.
#     print(loss.cpu().item())
#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step() # Run optimizer step.

def learning_rate_schedule(t, lr_max, lr_min, t_w, t_c):
    return t * lr_max / t_w if t < t_w else lr_min if t > t_c else lr_min + 0.5 * (1 + math.cos(math.pi * (t - t_w) / (t_c - t_w))) * (lr_max - lr_min)

def gradient_clipping(params :list[torch.nn.Parameter],max_l2_norm:float):
    
    g_l2_norm = math.sqrt(sum([p.grad.square().sum() for p in params if p.grad is not None]))
    factor = max_l2_norm/(g_l2_norm + 1e-6)
    
    if g_l2_norm >= max_l2_norm:
         for p in params:
             if p.grad is not None:
                 p.grad = p.grad*factor  
    return params   


