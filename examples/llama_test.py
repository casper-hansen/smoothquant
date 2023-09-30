import os
import torch
import numpy as np
from tqdm import tqdm
from smoothquant.smooth import smooth_lm
from smoothquant.calibration import get_act_scales
from transformers import AutoModelForCausalLM, AutoTokenizer
from smoothquant.utils import Perplexity, PerplexityV2, quantize_llama_model

# Variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_name = 'TheBloke/Llama-2-7b-chat-fp16'
dataset = 'mit-han-lab/pile-val-backup'
n_grid = 20
ratio = 1/n_grid
scales = np.arange(ratio, (n_grid*ratio), ratio, dtype=float)
num_samples = 512
seq_len = 512

print('Loading perplexity tokens...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
init_ppl = PerplexityV2(None, tokenizer)
tokens = tokenizer(init_ppl._text, truncation=False, return_tensors='pt').input_ids.to("cuda")
fp_16_ppl = 0
results = []

with torch.device("cuda"):
    model_fp16 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')

    ppl = PerplexityV2(model_fp16, tokenizer)
    out = ppl.calculate_perplexity(tokens=tokens)
    fp_16_ppl = out[-1]
    print("FP16 Perplexity:", fp_16_ppl)

for scale in scales:
    print('INT8 scale:', scale)
    with torch.device("cuda"):
        model_fp16 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')

        act_scales = get_act_scales(model_fp16, tokenizer, dataset, num_samples, seq_len)
        smooth_lm(model_fp16, act_scales, scale)
        model_int8 = quantize_llama_model(model_fp16, weight_quant="per_tensor", act_quant='per_tensor')

        ppl = PerplexityV2(model_int8, tokenizer)
        out = ppl.calculate_perplexity(tokens=tokens)

        regression_pct = ((out[-1]-fp_16_ppl)/fp_16_ppl)*100
        regression_str = 'higher' if regression_pct >= 0 else 'lower'
        result = f"INT8 Perplexity (Scale: {scale}): {out[-1]:.2f} ({regression_pct:.2f}% {regression_str} than FP16)"
        results.append(result)
        print(result)

print("FP16 Perplexity:", fp_16_ppl)
print('\n'.join(results))