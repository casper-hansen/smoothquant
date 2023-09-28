import os
import torch
import argparse
from smoothquant.calibration import get_act_scales
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='TheBloke/Llama-2-7b-chat-fp16', help='model name')
    parser.add_argument('--output-path', type=str, default='act_scales/llama-7b.pt',
                        help='where to save the act scales')
    parser.add_argument('--dataset-path', type=str, default='mit-han-lab/pile-val-backup',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--seq-len', type=int, default=512)
    args = parser.parse_args()
    return args

@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_name)

    act_scales = get_act_scales(model, tokenizer, args.dataset_path,
                                args.num_samples, args.seq_len)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)
    tokenizer.save_pretrained(args.output_path)

if __name__ == '__main__':
    main()
