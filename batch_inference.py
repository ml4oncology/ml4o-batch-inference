import argparse
import logging
import os
import pickle

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


ROOT_DIR = "/cluster/projects/gliugroup/2BLAST"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset in parquet format")
    parser.add_argument("--prompt-path", type=str, required=True, help="Path to the text file containing the system prompt")
    parser.add_argument("--output-path", type=str, help="Where to save the results")
    parser.add_argument("--model-name", type=str, default="Qwen3-14B")
    parser.add_argument("--text-col", type=str, default="note", help="Name of the column in the dataset containing the text")
    parser.add_argument("--id-col", type=str, default="note_id", help="Name of the column in the dataset containing the text identifier")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs used to split up the model weights for tensor parallelism. May not improve performance if model can fit in a single GPU with room to spare")
    parser.add_argument("--max-model-len", "--max-tokens", type=int, default=2048, help="Max number of tokens for prompt + output")
    parser.add_argument("--max-num-seqs", "--batch-size", type=int, default=256, help="Max number of prompts vLLM processes in parallel (higher = more throughput but more memory)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95, help="Fraction of GPU memory to use")
    parser.add_argument("--temperature", type=float, default=0.8, help=">1.0: More random/creative. 0.0: Greedy decoding. 1.0: standard randomness")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N batches")
    parser.add_argument("--resume-from-checkpoint", action="store_true", help="Resume from existing output file if it exists")
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = args.data_path.replace(".parquet", "_output.parquet")
        logger.info(f"No output path specified, using: {args.output_path}")
    checkpoint_path = args.output_path.replace(".parquet", "_checkpoint.parquet")

    # Load dataset and system instruction
    df = pd.read_parquet(args.data_path, columns=[args.id_col, args.text_col])
    with open(args.prompt_path, "r", encoding="utf-8") as file:
        system_instr = file.read()

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

    # Resume from checkpoint if exists
    if os.path.exists(checkpoint_path) and args.resume_from_checkpoint:
        with open(checkpoint_path, 'rb') as file:
            res = pickle.load(file)
        prompts = prompts[len(res):]
    else:
        res = []

    # Process the prompts in batches
    sampling_params = SamplingParams(temperature=args.temperature)
    for batch_num, i in enumerate(tqdm(range(0, len(prompts), args.max_num_seqs))):
        batch = prompts[i:i+args.max_num_seqs]
        # outputs = llm.chat(batch, sampling_params)
        outputs = llm.generate(batch, sampling_params)
        res += [{"prompt": output.prompt, "generated_output": output.outputs[0].text} for output in outputs]
        # save checkpoints at every Nth batch
        if batch_num % args.checkpoint_interval == 0:
            with open(checkpoint_path, "wb") as file:
                pickle.dump(res, file)

    # Save results
    res = pd.DataFrame(res, index=df[args.id_col])
    res.to_parquet(args.output_path, compression="zstd")
