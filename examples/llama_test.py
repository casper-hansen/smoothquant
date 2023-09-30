import os
import torch
from smoothquant.smooth import smooth_lm
from smoothquant.calibration import get_act_scales
from transformers import AutoModelForCausalLM, AutoTokenizer
from smoothquant.utils import Perplexity, PerplexityV2, quantize_llama_model

# Variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_name = 'TheBloke/Llama-2-7b-chat-fp16'
dataset = 'mit-han-lab/pile-val-backup'
scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
num_samples = 512
seq_len = 512

print('Loading perplexity tokens...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
init_ppl = Perplexity(None, tokenizer)
tokens = tokenizer(init_ppl._text, truncation=False, return_tensors='pt').input_ids.to("cuda")
all_perplexity = {}

with torch.device("cuda"):
    model_fp16 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')

    ppl = Perplexity(model_fp16, tokenizer)
    out = ppl.calculate_perplexity(tokens=tokens)
    all_perplexity["FP16 Perplexity (Scale: 1.0)"] = out[-1]

for scale in scales:
    with torch.device("cuda"):
        model_fp16 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')

        act_scales = get_act_scales(model_fp16, tokenizer, dataset, num_samples, seq_len)
        smooth_lm(model_fp16, act_scales, scale)
        model_int8 = quantize_llama_model(model_fp16, weight_quant='per_channel', act_quant='per_tensor')

        ppl = Perplexity(model_int8, tokenizer)
        out = ppl.calculate_perplexity(tokens=tokens)
        all_perplexity[f"INT8 Perplexity (Scale: {scale})"] = out[-1]

for i, (key, value) in enumerate(all_perplexity.items()):
    if i > 0:
        regression = all_perplexity["FP16 Perplexity (Scale: 1.0)"] / value * 100
    else:
        regression = 0
    
    regression_str = 'higher' if regression >= 0 else 'lower'

    print(f"{key}: {value:.2f} ({regression:.2f}% {regression_str} than FP16)")
