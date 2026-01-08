"""
Centralized inference pipeline using vLLM.

References
----------
https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/prefix_caching.py
https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/structured_outputs.py
"""
import argparse
import gc
import json
import logging
import os

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


ROOT_DIR = "/cluster/projects/gliugroup/2BLAST"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the parquet dataset")
    parser.add_argument("--prompt-path", type=str, required=True, help="Path to the text file containing the system prompt")
    parser.add_argument("--output-path", type=str, required=True, help="Directory to save the results")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Path to the tokenizer")
    parser.add_argument("--json-struct-path", type=str, default=None, help="Path to the JSON file describing the desired structured output")
    parser.add_argument("--model-name", type=str, default="Qwen3-14B")
    parser.add_argument("--text-col", type=str, default="note", help="Name of the column in the dataset containing the text")
    parser.add_argument("--id-col", type=str, default="note_id", help="Name of the column in the dataset containing the text identifier")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs used to split up the model weights for tensor parallelism. May not improve performance if model can fit in a single GPU with room to spare")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max number of tokens for prompt + output")
    parser.add_argument("--max-output-len", "--max-tokens", type=int, default=512, help="Max number of tokens for output")
    parser.add_argument("--max-num-seqs", "--batch-size", type=int, default=16, help="Max number of prompts vLLM processes in parallel (higher = more throughput but more memory)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95, help="Fraction of GPU memory to use")
    parser.add_argument("--temperature", type=float, default=0.8, help=">1.0: More random/creative. 0.0: Greedy decoding. 1.0: standard randomness")
    parser.add_argument("--enable-thinking", action="store_true", help="Whether to enable thinking (may use up a lot of output tokens)")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Save checkpoint every N batches")
    parser.add_argument("--resume-from-checkpoint", action="store_true", help="Resume from existing output directory if it exists")
    args = parser.parse_args()

    os.makedirs(f"{args.output_path}/generated_output", exist_ok=True)
    model_path = f"{ROOT_DIR}/LLMs/{args.model_name}"
    if args.tokenizer_path is None:
        args.tokenizer_path = model_path

    # Load dataset and system instruction
    df = pd.read_parquet(args.data_path, columns=[args.id_col, args.text_col])
    with open(args.prompt_path, "r", encoding="utf-8") as file:
        system_instr = file.read()

    # Load structured output schema if provided
    json_schema = None
    if args.json_struct_path is not None:
        with open(args.json_struct_path, "r") as f:
            json_schema = json.load(f)
        logger.info("Structured JSON output enabled")

    # Validate metadata on resume
    metadata_path = f"{args.output_path}/metadata.json"
    current_metadata = vars(args)
    if args.resume_from_checkpoint:
        with open(metadata_path, "r") as f:
            saved_metadata = json.load(f)
            saved_metadata["resume_from_checkpoint"] = True
        # Compare saved metadata with current args
        assert saved_metadata == current_metadata, (
            f"Cannot resume: metadata mismatch detected.\n"
            f"Saved: {saved_metadata}\n"
            f"Current: {current_metadata}"
        )
    else:
        with open(metadata_path, "w") as f:
            json.dump(current_metadata, f, indent=2)

    # Ignore prompts that's too long (exceeds max-model-len minus max-output-len)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    system_instr_token_length = tokenizer(system_instr, add_special_tokens=False, return_length=True)["length"]
    token_lengths, bs = [], int(1e5)
    for i in range(0, len(df), bs):
        token_lengths += tokenizer(df[args.text_col].iloc[i:i+bs].tolist(), add_special_tokens=False, return_length=True)["length"]
    df["token_length"] = token_lengths
    mask = system_instr_token_length + df["token_length"] > args.max_model_len - args.max_output_len
    if mask.any():
        df, df_ignored = df[~mask], df[mask]
        logger.info(f"Ignoring {mask.sum()} ({mask.mean()*100:.3f}%) samples that exceed {args.max_model_len-args.max_output_len} tokens")
        df_ignored.to_parquet(f"{args.output_path}/unprocessed.parquet", index=False, compression="gzip")
    # Tokenizer uses a lot of memory, free it up
    del tokenizer
    gc.collect()

    # Initialize vLLM
    llm = LLM(
        model=model_path,
        tokenizer=args.tokenizer_path,
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
    # NOTE: if you don't apply chat template, you won't get a structured thinking
    prompts = [
        [{"role": "system", "content": system_instr},
         {"role": "user", "content": text}]
         for text in df[args.text_col]
    ]
    # prompts = [f"{system_instr}\n{text}" for text in df[args.text_col]]

    # Create sampling parameter object
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_output_len)
    if json_schema is not None:
        # Ensure sturctured JSON output
        structured_outputs_params_json = StructuredOutputsParams(json=json_schema)
        sampling_params.structured_outputs = structured_outputs_params_json

    # Warmup so that the shared prompt's KV cache is computed (for prefix caching)
    llm.chat(prompts[0], sampling_params)
    # llm.generate(prompts[0], sampling_params)

    # Process the prompts in batches
    batch_size = args.max_num_seqs * args.checkpoint_interval
    for batch_num, i in enumerate(tqdm(range(0, len(prompts), batch_size))):
        # Check if already processed
        batch_filepath = f"{args.output_path}/generated_output/batch_{batch_num}.parquet"
        if args.resume_from_checkpoint and os.path.exists(batch_filepath):
            logger.info(f"Skipping batch {batch_num}")
            continue

        # Process them
        batch = prompts[i:i+batch_size]
        ids = df[args.id_col].iloc[i:i+batch_size]
        outputs = llm.chat(batch, sampling_params, chat_template_kwargs={"enable_thinking": args.enable_thinking})
        res = [{"prompt": output.prompt, "generated_output": output.outputs[0].text} for output in outputs]

        # Save results
        res = pd.DataFrame(res, index=ids)
        res.to_parquet(batch_filepath, compression="zstd")
