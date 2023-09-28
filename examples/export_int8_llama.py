import torch
import argparse
from pathlib import Path
from smoothquant.smooth import smooth_lm
from smoothquant.llama import Int8LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from smoothquant.calibration import get_static_llama_decoder_layer_scales

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default='TheBloke/Llama-2-7b-chat-fp16')
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--act-scales", type=str,
                        default='act_scales/llama-7b.pt')
    parser.add_argument("--output-path", type=str, default='int8_models')
    parser.add_argument('--dataset-path', type=str, default='mit-han-lab/pile-val-backup',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--export-FT', default=False, action="store_true")
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    act_scales = torch.load(args.act_scales)
    smooth_lm(model, act_scales, 0.5)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    decoder_layer_scales, raw_scales = get_static_llama_decoder_layer_scales(model,
                                                                            tokenizer,
                                                                            args.dataset_path,
                                                                            num_samples=args.num_samples,
                                                                            seq_len=args.seq_len)
    output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant.pt")
    if args.export_FT:
        model.save_pretrained(output_path)
        print(f"Saved smoothed model at {output_path}")

        output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant-scales.pt")
        torch.save(raw_scales, output_path)
        print(f"Saved scaling factors at {output_path}")
    else:
        int8_model = Int8LlamaForCausalLM.from_float(model, decoder_layer_scales)
        int8_model.save_pretrained(output_path)
        print(f"Saved int8 model at {output_path}")