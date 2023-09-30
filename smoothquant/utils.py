import sys
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from smoothquant.fake_quant import W8A8Linear
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

def quantize_llama_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
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

class PerplexityV2:
    def __init__(self, model, tokenizer, dataset_path='wikitext', dataset_name=None, split='test', text_column='text'):
        self._model = model
        self._tokenizer = tokenizer
        self._dataset_path = dataset_path
        self._dataset_name = dataset_name
        self._split = split
        self._text_column = text_column
        self._text = self._prepare_data()
        self._total_nll = 0.0
        self._count = 0
        self.max_length = 2048

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

    def calculate_perplexity(self, stride=512, tokens=None, n_rounds=-1):
        if tokens is None:
            tokens = self._tokenizer(self._text, truncation=False, return_tensors="pt").input_ids.to(self._model.device)
        
        seq_len = tokens.size(1)
        prev_end_loc = 0
        all_perplexity = []

        if n_rounds != -1:
            full_range = [list(range(0, seq_len, stride))[:n_rounds]]
        else:
            full_range = range(0, seq_len, stride)

        with tqdm(full_range, desc="Perplexity: - ") as progress:
            for begin_loc in progress:
                nll = self._process_batch(tokens, begin_loc, prev_end_loc)
                self._total_nll += nll.item()
                self._count += 1
                prev_end_loc = min(begin_loc + self.max_length, seq_len)

                # Compute the running perplexity and update the tqdm description
                running_perplexity = np.exp(self._total_nll / self._count)
                all_perplexity.append(running_perplexity)
                progress.set_description(f"Perplexity: {running_perplexity:.4f}")

        return all_perplexity

    def _process_batch(self, tokens, begin_loc, prev_end_loc):
        end_loc = min(begin_loc + self.max_length, tokens.size(1))
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = tokens[:, begin_loc:end_loc].to(self._model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = self._model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        return neg_log_likelihood