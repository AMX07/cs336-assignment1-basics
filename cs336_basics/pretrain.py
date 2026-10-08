
import argparse
import string
import torch
from cs336_basics.gpt import *
import numpy as np
import os
from dataclasses import dataclass, fields


@dataclass
class hyperparams:
    batch_size: int = 32
    vocab_size: int =50257
    context_length: int =128
    num_layers: int=2
    d_model: int=64
    num_heads: int =4
    d_ff: int=4288
    rope_theta: float=10000.0
    dataset: str = "/Users/anshmittal/Documents/cs336/assignment1-basics/out/owt_train.npy"
    val_dataset: str = "/Users/anshmittal/Documents/cs336/assignment1-basics/out/owt_valid.npy"
    t: int = 2000
    eval_every : int = 1000 #t%2
    t_w: int = 5
    lr_max:float = 1e-3
    lr_min : float = 1e-4
    t_c: int = 6
    device : str = "mps"
    save_every : int = t / 10
    eval_batches : int = 10

def cross_entropy(inputs, targets):
    # subtracting the max values
    shifted = inputs - inputs[-1].max()
    # torch.Tensor
    log_sum_exp = shifted.exp().sum(dim=-1).log()
    target_logits = shifted.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)

    return (log_sum_exp - target_logits).mean()

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
        optimizer step. We dont use it but we are passing it so as to comply with the API
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
#weights = model.parameters()
# opt = SGD([weights], lr=1e3) #1e1

# for t in range(10):
#     opt.zero_grad() # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.
#     print(loss.cpu().item())
#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step() # Run optimizer step.

def learning_rate_schedule(t, lr_max, lr_min, t_w, t_c):
    '''
    t: current iteration
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    

    '''
    return t * lr_max / t_w if t < t_w else lr_min if t > t_c else lr_min + 0.5 * (1 + math.cos(math.pi * (t - t_w) / (t_c - t_w))) * (lr_max - lr_min)

def gradient_clipping(params :list[torch.nn.Parameter],max_l2_norm:float):
     
    g_l2_norm = math.sqrt(sum([p.grad.square().sum() for p in params if p.grad is not None]))
    factor = max_l2_norm/(g_l2_norm + 1e-6)
    
    if g_l2_norm >= max_l2_norm:
         for p in params:
             if p.grad is not None:
                 p.grad = p.grad*factor  
    return params   

import numpy.typing as npt

def data_loader(
    dataset: npt.NDArray, batch_size: int, context_length: int, device=torch.device('mps')
) -> tuple[torch.Tensor, torch.Tensor]:
    
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    using np to allow lazy loading the data using np.memmap when the dataset is too big to load into memory 

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset. (x1, …, xn)
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
   
    # dataset =  (dataset)
    idx = torch.randint(0,len(dataset)-context_length,(batch_size,))
    x = torch.tensor([dataset[i:i+context_length] for i in idx],dtype=torch.long,device=device)
    y = torch.tensor([dataset[i+1:i+context_length+1] for i in idx],dtype=torch.long,device=device)
    
    return x,y

def save_checkpoint(model, optimizer, iteration, out):
    '''
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    iteration: int
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    
    state_dict method of both the model and the optimizer to get their relevant states 
    torch.save(obj, out) to dump obj into 'out', PyTorch supports path or a file-like object. 
    typically obj is a dict
    '''
    # saving multiple checkpoints
    obj =  {"iteration":iteration, "model_state_dict":model.state_dict(), "optim_state_dict":optimizer.state_dict()}
    
    # torch.optim.Optimizer.state_dict()
    
    torch.save(obj, out)
    
def load_checkpoint(src, model, optimizer):
    '''should load a checkpoint from src (path or file-like object) 
    and then recover the model and optimizer states from that checkpoint. Your function
    should return the iteration number that was saved to the checkpoint. You can use
    torch.load(src) to recover what you saved in your save_checkpoint implementation, and the
    load_state_dict method in both the model and optimizer to return them to their previous
    states.
    This function expects the following parameters:
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    Implement the [adapters.run_save_checkpoint] and [adapters.run_load_checkpoint] adapters,
    and make sure they pass uv run pytest -k test_checkpointing.
    '''

    obj = torch.load(src) #{iteration,model.state_dict(), optimizer.state_dict()}
    # obj =  {"iteration":iteration,"model.state_dict()":model.state_dict(), "optimizer.state_dict()":optimizer.state_dict()}

    iteration, model_state_dict, optim_state_dict = obj.values()

    # torch.nn.Module().load_state_dict(model_state_dict)
    model.load_state_dict(model_state_dict)

    # torch.optim.Optimizer().load_state_dict(optim_state_dict)
    optimizer.load_state_dict(optim_state_dict)

    return iteration

def main(cfg) :
    
    '''
    training loop
    '''
    model = transformer_lm(cfg.vocab_size,cfg.context_length,cfg.num_layers,cfg.d_model,cfg.num_heads,cfg.d_ff,cfg.rope_theta) #todo: redesign to prevent repeated arguments
    model.to(device=cfg.device)
    
    dataset = np.load(cfg.dataset, mmap_mode="r")
    # dataset = np.memmap(filename=cfg.dataset, dtype=np.uint16, mode="r")
    # params = 
    # lr = learning_rate_schedule(), need to know max itertions
    opt = adamw(model.parameters(), lr=cfg.lr_max)
    val_dataset = np.load(cfg.val_dataset, mmap_mode="r")
    losses = []
                

    
    for t in range(cfg.t):
        lr = learning_rate_schedule(t, cfg.lr_max, cfg.lr_min, cfg.t_w, cfg.t_c)

        for group in opt.param_groups:
            group["lr"] = lr
        # load tokenized data
        x,y = data_loader(dataset, cfg.batch_size, cfg.context_length, cfg.device)# x,y = tok_embd(x), tok_embd(y)
        # print(x.dtype,y.dtype)
        x = model(x,cfg.num_heads,cfg.d_model) # in_indices,num_heads,d_model 
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        # calculate loss by comparing the outputs from the model vs targets
        loss = cross_entropy(x,y) # Compute a scalar loss value.
        # print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step
        # what's the diff between lr used in adawm class and lr schedule we implemented, how to use which one?  
        
        
        ckpt_dir = f"checkpoints/{run.name}"
        os.makedirs(ckpt_dir, exist_ok=True)

        if (t + 1) % cfg.save_every == 0:
            losses.append(loss)
            run.log({"loss": sum(losses) / len(losses)}, step=t + 1)
            save_checkpoint(model, opt, t + 1, f"{ckpt_dir}/ckpt_{t + 1}.pt")

        
        if (t + 1) % cfg.eval_every == 0:
            with torch.no_grad():
                val_losses = []
                for _ in range(cfg.eval_batches):
                    vx, vy = data_loader(val_dataset, cfg.batch_size, cfg.context_length, cfg.device)
                    val_losses.append(cross_entropy(model(vx, cfg.num_heads, cfg.d_model), vy).item())
            run.log({"val_loss": sum(val_losses) / len(val_losses)}, step=t + 1)
                    

                    
    
    save_checkpoint(model, opt, cfg.t, f"{ckpt_dir}/ckpt_final.pt")
    run.finish()
    
    # try to actually run the loop


def decoding(prompt):
    """
    1. load the model checkpoints
    prompt = T,C
    """
    # forward transformer lm with batch size = 1

    model = transformer_lm(cfg)



import random
import wandb
 
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="hyperparams and configs for training")
    
    for hyperparam in fields(hyperparams):
        parser.add_argument("--" + hyperparam.name, type= hyperparam.type, default=hyperparam.default)

    args = vars(parser.parse_args())

    run = wandb.init(
        entity="alephnott",
        project="assignment1",
        config=hyperparams(**args),
    )
    # main(hyperparams(**args))
    decoding(hyperparams(**args))



