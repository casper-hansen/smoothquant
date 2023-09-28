import sys
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm_notebook as tqdm
from smoothquant.fake_quant import W8A8Linear
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    for name, m in model.model.named_modules():
        if isinstance(m, LlamaAttention):
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.o_proj = W8A8Linear.from_float(m.o_proj, weight_quant=weight_quant, act_quant=act_quant)
        
        elif isinstance(m, LlamaMLP):
            m.gate_proj = W8A8Linear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant)
            m.up_proj = W8A8Linear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant)
            m.down_proj = W8A8Linear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant)
            
    return model

class Perplexity:
    def __init__(self, model, tokenizer, dataset_path='wikitext', dataset_name=None, split='test', text_column='text'):
        self._model = model
        self._tokenizer = tokenizer
        self._dataset_path = dataset_path
        self._dataset_name = dataset_name
        self._split = split
        self._text_column = text_column
        self._text = self._prepare_data()
    
    def _get_device(self):
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda:0'
        else:
            return 'cpu'
    
    def _prepare_data(self):
        if self._dataset_path == 'wikitext':
            self._dataset_name = 'wikitext-2-raw-v1'
        
        data = load_dataset(self._dataset_path, self._dataset_name, split=self._split)
        text_list = [' \n' if s == '' else s for s in data[self._text_column]]
        return ''.join(text_list)

    @staticmethod
    def softmax(logits):
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum(axis=0)

    def calculate_perplexity(self, n_ctx=512, n_batch=512, tokens=None):
        self._tokenizer.model_max_length = sys.maxsize
        if tokens is None:
            tokens = self._tokenizer(self._text, truncation=False, return_tensors='pt').input_ids.to(self._model.device)

        nll = 0.0
        count = 0
        curr_ppl = 0
        all_perplexity = []

        with tqdm(range(len(tokens[0]) // n_ctx), desc="Perplexity: - ") as progress:
            for i in progress:
                nll, count = self._process_batch(i, n_ctx, n_batch, tokens, nll, count)
                curr_ppl = np.exp(nll / count)
                all_perplexity.append(curr_ppl)
                progress.set_description(f"Perplexity: {curr_ppl:.4f}")

        return all_perplexity

    def _process_batch(self, i, n_ctx, n_batch, tokens, nll, count):
        start = i * n_ctx
        end = start + n_ctx
        num_batches = (n_ctx + n_batch - 1) // n_batch
        logits = []

        for j in range(num_batches):
            batch_start = start + j * n_batch
            batch_size = min(end - batch_start, n_batch)
            token_org = tokens[0][batch_start].item()

            if j == 0:
                tokens[0][batch_start] = self._tokenizer.bos_token_id

            batch_logits = self._compute_batch_logits(tokens, batch_start, batch_size)
            tokens[0][batch_start] = token_org
            logits.append(batch_logits)

        for j in range(min(512, n_ctx // 2), n_ctx - 1):
            tok_logits = logits[0][0][j].cpu().numpy()
            prob = self.softmax(tok_logits)[tokens[0][start + j + 1]]
            nll += -np.log(prob, where=prob>0)
            count += 1

        return nll, count

    def _compute_batch_logits(self, tokens, batch_start, batch_size):
        with torch.no_grad():
            outputs = self._model(tokens[:, batch_start:batch_start+batch_size])
        return outputs.logits.detach()
