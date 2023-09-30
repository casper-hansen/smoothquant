import os
import torch
from smoothquant.smooth import smooth_lm
from smoothquant.calibration import get_act_scales
from transformers import AutoModelForCausalLM, AutoTokenizer
from smoothquant.utils import Perplexity, PerplexityV2, quantize_llama_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print('Loading perplexity tokens...')
tokenizer = AutoTokenizer.from_pretrained('TheBloke/Llama-2-7b-chat-fp16')
init_ppl = Perplexity(None, tokenizer)
tokens = tokenizer(init_ppl._text, truncation=False, return_tensors='pt').input_ids.to("cuda")

with torch.device("cuda"):
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        'TheBloke/Llama-2-7b-chat-fp16', torch_dtype=torch.float16, device_map='auto')

    ppl = Perplexity(model_fp16, tokenizer)
    out = ppl.calculate_perplexity(tokens=tokens)
    print(f'FP16 perplexity (scale: 1.0): {out[-1]}')

for scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    with torch.device("cuda"):
        model_fp16 = AutoModelForCausalLM.from_pretrained(
            'TheBloke/Llama-2-7b-chat-fp16', torch_dtype=torch.float16, device_map='auto')

        act_scales = get_act_scales(model_fp16, tokenizer, 'mit-han-lab/pile-val-backup', 512, 512)
        smooth_lm(model_fp16, act_scales, scale)
        model_int8 = quantize_llama_model(model_fp16, weight_quant='per_tensor', act_quant='per_tensor')

        ppl = Perplexity(model_int8, tokenizer)
        out = ppl.calculate_perplexity(tokens=tokens)
        print(f'INT8 perplexity (scale: {scale}): {out[-1]}')

# Perplexity (v1 not v2)
# FP16 perplexity (scale: 1.0): 7.627832972676341
# INT8 perplexity (scale: 0.1): 21.436434377211963
# INT8 perplexity (scale: 0.2): 19.824321944984888
# INT8 perplexity (scale: 0.3): 20.036439254124588
# INT8 perplexity (scale: 0.4): 20.063201060778677
# INT8 perplexity (scale: 0.5): 19.75134567944594
# INT8 perplexity (scale: 0.6): 20.19175300029989
# INT8 perplexity (scale: 0.7): 19.92419387177099
# INT8 perplexity (scale: 0.8): 22.096610772575158
# INT8 perplexity (scale: 0.9): 35.50489966108907