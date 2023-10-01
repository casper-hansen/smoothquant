import os
import torch
import numpy as np
from copy import deepcopy
from smoothquant.smooth import smooth_lm
from smoothquant.calibration import get_act_scales
from transformers import AutoModelForCausalLM, AutoTokenizer
from smoothquant.utils import PerplexityV2, quantize_llama_model

# Variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_name = 'PY007/TinyLlama-1.1B-Chat-v0.2'
dataset = 'mit-han-lab/pile-val-backup'
n_grid = 20
ratio = 1/n_grid
# scales = np.arange(ratio, (n_grid*ratio)+ratio, ratio, dtype=float)[::-1]
scales = [0.4, 0.45, 0.5, 0.55, 0.6]
num_samples = 512
seq_len = 512
weight_quant = "per_tensor"
act_quant = "per_tensor"

print('Loading perplexity tokens...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
init_ppl = PerplexityV2(None, tokenizer)
tokens = tokenizer(init_ppl._text, truncation=False, return_tensors='pt').input_ids.to("cuda")
fp_16_ppl = 0
results = []
act_scales = None

# FP16
with torch.device("cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')

    ppl = PerplexityV2(model, tokenizer)
    out = ppl.calculate_perplexity(tokens=tokens)
    fp_16_ppl = out[-1]
    print("FP16 Perplexity:", fp_16_ppl)

# W8A8
for scale in scales:
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')

        if act_scales is None:
            act_scales = get_act_scales(model, tokenizer, dataset, num_samples, seq_len)

        # quantize
        smooth_lm(model, deepcopy(act_scales), scale)
        model = quantize_llama_model(model, weight_quant=weight_quant, act_quant=act_quant)

        # compute perplexity
        ppl = PerplexityV2(model, tokenizer)
        out = ppl.calculate_perplexity(tokens=tokens)

        regression_pct = ((out[-1]-fp_16_ppl)/fp_16_ppl)*100
        regression_str = 'higher' if regression_pct >= 0 else 'lower'
        result = f"INT8 Perplexity (Scale: {scale}): {out[-1]:.2f} ({regression_pct:.2f}% {regression_str} than FP16)"
        results.append(result)
        print(result)

print("FP16 Perplexity:", fp_16_ppl)
print('\n'.join(results))