import argparse
import json
import logging
import os

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


ROOT_DIR = "/cluster/projects/gliugroup/2BLAST"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the parquet dataset")
    parser.add_argument("--prompt-path", type=str, required=True, help="Path to the text file containing the system prompt")
    parser.add_argument("--output-path", type=str, required=True, help="Directory to save the results")
    parser.add_argument("--model-name", type=str, default="Qwen3-14B")
    parser.add_argument("--text-col", type=str, default="note", help="Name of the column in the dataset containing the text")
    parser.add_argument("--id-col", type=str, default="note_id", help="Name of the column in the dataset containing the text identifier")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs used to split up the model weights for tensor parallelism. May not improve performance if model can fit in a single GPU with room to spare")
    parser.add_argument("--max-model-len", "--max-tokens", type=int, default=2048, help="Max number of tokens for prompt + output")
    parser.add_argument("--max-num-seqs", "--batch-size", type=int, default=256, help="Max number of prompts vLLM processes in parallel (higher = more throughput but more memory)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95, help="Fraction of GPU memory to use")
    parser.add_argument("--temperature", type=float, default=0.8, help=">1.0: More random/creative. 0.0: Greedy decoding. 1.0: standard randomness")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Save checkpoint every N batches")
    parser.add_argument("--resume-from-checkpoint", action="store_true", help="Resume from existing output directory if it exists")
    args = parser.parse_args()

    # Load dataset and system instruction
    df = pd.read_parquet(args.data_path, columns=[args.id_col, args.text_col])
    with open(args.prompt_path, "r", encoding="utf-8") as file:
        system_instr = file.read()

    # Validate metadata on resume
    metadata_path = f"{args.output_path}/metadata.json"
    current_metadata = vars(args)
    if args.resume_from_checkpoint:
        with open(metadata_path, "r") as f:
            saved_metadata = json.load(f)
        # Compare saved metadata with current args
        assert saved_metadata == current_metadata, (
            f"Cannot resume: metadata mismatch detected.\n"
            f"Saved: {saved_metadata}\n"
            f"Current: {current_metadata}"
        )
    else:
        with open(metadata_path, "w") as f:
            json.dump(current_metadata, f, indent=2)

    # Initialize vLLM
    llm = LLM(
        model=f"{ROOT_DIR}/LLMs/{args.model_name}",
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,

        # pre-allocates KV cache memory based on max-model-len, regardless of actual
        # prompt length. A large max-model-len will eat up a large chunk of GPU memory,
        # leaving little memory for the actual inputs/outputs
        max_model_len=args.max_model_len,

        # allocates a much larger KV cache memory, can be used dynamically for input
        # tokens and output tokens and cached prefix blocks. Do not worry if most of the
        # GPU RAM is taken, it already includes the memory for inputs/outputs
        enable_prefix_caching=True,
    )

    # Create the prompts
    # prompts = [
    #     [{"role": "system", "content": system_instr},
    #      {"role": "user", "content": text}]
    #      for text in df[args.text_col]
    # ]
    prompts = [f"{system_instr}/n{text}" for text in df[args.text_col]]

    # Process the prompts in batches
    batch_size = args.max_num_seqs * args.checkpoint_interval
    sampling_params = SamplingParams(temperature=args.temperature)
    for batch_num, i in enumerate(tqdm(range(0, len(prompts), batch_size))):
        batch = prompts[i:i+batch_size]
        # outputs = llm.chat(batch, sampling_params)
        outputs = llm.generate(batch, sampling_params)
        res = [{"prompt": output.prompt, "generated_output": output.outputs[0].text} for output in outputs]

        # save results
        res = pd.DataFrame(res, index=df[args.id_col])
        res.to_parquet(f"{args.output_path}/batch_{batch_num}.parquet", compression="zstd")
