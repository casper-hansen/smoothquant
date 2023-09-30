import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear
import os
import gc
from torch.nn.functional import pad
import torch
from smoothquant.smooth import smooth_lm
from smoothquant.utils import Perplexity
from smoothquant.calibration import get_act_scales

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Evaluator:
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latency = 0
        for batch in self.dataset:
            input_ids = batch['input_ids'].cuda().unsqueeze(0)
            label = input_ids[:, -1]
            pad_len = 512 - input_ids.shape[1]
            input_ids = pad(input_ids, (0, pad_len), value=1)
            torch.cuda.synchronize()
            start.record()
            outputs = model(input_ids)
            end.record()
            torch.cuda.synchronize()
            latency += start.elapsed_time(end)
            last_token_logits = outputs.logits[:, -2-pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()

        acc = hit / total
        lantecy = latency / len(self.dataset)
        return acc, lantecy

def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)
    return model

def print_model_size(model):
    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}MB'.format(size_all_mb))

from datasets import load_dataset
dataset = load_dataset('lambada', split='validation[:1000]')

tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-6.7b')
evaluator = Evaluator(dataset, tokenizer)

# fp16 opt-6.7b
with torch.device("cuda"):
    model_fp16 = OPTForCausalLM.from_pretrained(
        'facebook/opt-6.7b', torch_dtype=torch.float16, device_map='auto')

act_scales = get_act_scales(model_fp16, tokenizer, 'mit-han-lab/pile-val-backup', 512, 512)
smooth_lm(model_fp16, act_scales, 0.5)
model_int8 = quantize_model(model_fp16)

# acc, latency = evaluator.evaluate(model_int8)
# print(f'OPT-6.7B accuracy: {acc}, per-sample lantecy: {latency:.3f}ms')
# Prints: OPT-6.7B accuracy: 0.791, per-sample lantecy: 101.201ms

ppl = Perplexity(model_int8, tokenizer)
out = ppl.calculate_perplexity()
print(f'INT8 perplexity: {out[-1]}')
# Prints: 11.0432