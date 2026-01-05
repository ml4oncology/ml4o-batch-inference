"""
Distributed inference pipeline using vLLM and Ray Data framework for orchestration.

Ray Data framework supports:
- ✅ Continuous batching - keeps GPUs saturated by dynamically feeding them prompts
- ✅ Automatic load balancing - Ray handles distribution, slower GPUs get fewer prompts
- ✅ Streaming execution - can process datasets larger than cluster RAM
- ✅ Fault tolerance - built-in retry semantics
- ✅ Better GPU utilization - adaptive batching maximizes throughput
- ✅ Scalability - add more GPUs without code changes (just adjust concurrency)
- ✅ Production features - read/write to cloud storage, multiple file formats

References
----------
https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/batch_llm_inference.py
"""
import argparse
import gc
import json
import logging

import ray
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
from transformers import AutoTokenizer


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
    parser.add_argument("--max-num-seqs", "--batch-size", type=int, default=256, help="Max number of prompts vLLM processes in parallel (higher = more throughput but more memory)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95, help="Fraction of GPU memory to use")
    parser.add_argument("--temperature", type=float, default=0.8, help=">1.0: More random/creative. 0.0: Greedy decoding. 1.0: standard randomness")
    parser.add_argument("--enable-thinking", action="store_true", help="Whether to enable thinking (may use up a lot of output tokens)")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of parallel vLLM replicas for Ray Data (more replicas = more GPUs utilized)")
    args = parser.parse_args()

    model_path = f"{ROOT_DIR}/LLMs/{args.model_name}"
    if args.tokenizer_path is None:
        args.tokenizer_path = model_path

    # Load dataset and system instruction
    ds = ray.data.read_parquet(args.data_path, columns=[args.id_col, args.text_col])
    with open(args.prompt_path, "r", encoding="utf-8") as file:
        system_instr = file.read()

    # Ignore prompts that's too long (exceeds max-model-len minus max-output-len)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    sys_token_len = tokenizer(system_instr, add_special_tokens=False, return_length=True)["length"]
    max_input_tokens = args.max_model_len - args.max_output_len
    original_count = ds.count()
    ds = ds.filter(lambda row: sys_token_len + tokenizer(row[args.text_col], add_special_tokens=False, return_length=True)["length"] <= max_input_tokens)
    filtered_count = ds.count()
    if original_count > filtered_count:
        logger.info(f"Ignoring {original_count - filtered_count} ({(original_count - filtered_count)/original_count*100:.3f}%) samples exceeding {max_input_tokens} tokens")
    # Tokenizer uses a lot of memory, free it up
    del tokenizer
    gc.collect()

    # Load structured output schema if provided
    json_schema = None
    if args.json_struct_path is not None:
        with open(args.json_struct_path, "r") as f:
            json_schema = json.load(f)
        logger.info("Structured JSON output enabled")

    # Save metadata
    metadata_path = f"{args.output_path}/metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Configure vLLM engine for Ray Data
    config = vLLMEngineProcessorConfig(
        model_source=model_path,
        engine_kwargs={
            "tokenizer": args.tokenizer_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_prefix_caching": True,  # For shared system prompt optimization
            "enable_chunked_prefill": True,  # Ray Data optimization
        },
        concurrency=args.concurrency,  # Number of parallel vLLM replicas
        batch_size=args.max_num_seqs,  # Prompts per batch
        chat_template_kwargs={"enable_thinking": args.enable_thinking},
    )

    # Define preprocessing function
    def preprocess_row(row):
        """Format each row into chat messages and sampling parameters."""
        messages = [
            {"role": "system", "content": system_instr},
            {"role": "user", "content": row[args.text_col]},
        ]
        sampling_params = {"temperature": args.temperature, "max_tokens": args.max_output_len}
        if json_schema is not None:
            sampling_params["guided_json"] = json_schema
        return {
            "messages": messages,
            "sampling_params": sampling_params,
            args.id_col: row[args.id_col],
        }

    # Define postprocessing function
    def postprocess_row(row):
        """Extract generated text and preserve metadata."""
        return {
            args.id_col: row[args.id_col],
            "messages": row["messages"],
            "generated_output": row["generated_text"],
        }

    # Build vLLM processor with Ray Data
    logger.info("Building vLLM processor with Ray Data framework")
    vllm_processor = build_llm_processor(
        config,
        preprocess=preprocess_row,
        postprocess=postprocess_row,
    )

    # Execute inference using Ray Data
    logger.info("Starting batch inference")
    result_ds = vllm_processor(ds)

    # Write results to parquet
    logger.info(f"Writing results to {args.output_path}")
    result_ds.write_parquet(
        args.output_path,
        compression="zstd",
    )

    logger.info("Batch inference completed successfully")
