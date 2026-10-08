
'''
Total_Flops_forward = Batch_size * (num_layes * transformer_block_flops + LM_head_flops)

LM_head_flops = output_layer_flops = 2*seq_len*

transformer_block_flops:
1.MHA: q,k,v projections + attention computation + output_projection
    = (3 * (2(Seq_len)(d_m)^2) +( 2* 2(seq_len)^2*(head_size) )+ 2(seq_len)(d_m)^2

2.swiglu_ffn: 2* 2(seq_len)(d_m)(d_ff) + 2(seq_len)(d_ff)(d_m)

'''

# model = transformer_lm(
#     vocab_size=50257,
#     context_length=1024,
#     num_layers=48,
#     d_model=1600,
#     num_heads=25,
#     d_ff=4288,
#     rope_theta=10000.0,
# )

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

