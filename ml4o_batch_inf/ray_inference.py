"""
Distributed inference pipeline using vLLM and Ray Data framework for orchestration.

Ray Data framework supports:
- Continuous batching - keeps GPUs saturated by dynamically feeding them prompts
- Automatic load balancing - Ray handles distribution, slower GPUs get fewer prompts
- Streaming execution - can process datasets larger than cluster RAM
- Fault tolerance - built-in retry semantics
- Better GPU utilization - adaptive batching maximizes throughput
- Scalability - add more GPUs without code changes (just adjust concurrency)
- Production features - read/write to cloud storage, multiple file formats

References
----------
https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/batch_llm_inference.py
"""
import gc
import json
import logging
import os

import pandas as pd
import ray
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
from transformers import AutoTokenizer
from vllm.sampling_params import StructuredOutputsParams


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


ROOT_DIR = "/cluster/projects/gliugroup/2BLAST"


def run(
    data_path: str,
    prompt_path: str,
    output_path: str,
    json_struct_path: str | None,
    model_name: str,
    text_col: str,
    id_col: str,
    tensor_parallel_size: int,
    max_model_len: int,
    max_output_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    temperature: float,
    enable_thinking: bool,
    checkpoint_interval: int,
    resume_from_checkpoint: bool,
    concurrency: int,
) -> None:
    """Run distributed batch inference using vLLM + Ray Data."""
    os.makedirs(f"{output_path}/generated_output", exist_ok=True)
    model_path = f"{ROOT_DIR}/LLMs/{model_name}"

    # Load dataset and system instruction
    df = pd.read_parquet(data_path, columns=[id_col, text_col])
    with open(prompt_path, "r", encoding="utf-8") as file:
        system_instr = file.read()

    # Load structured output schema if provided
    json_schema = None
    if json_struct_path is not None:
        with open(json_struct_path, "r") as f:
            json_schema = json.load(f)
        logger.info("Structured JSON output enabled")

    # Validate metadata on resume
    metadata_path = f"{output_path}/metadata.json"
    current_metadata = {
        "data_path": data_path,
        "prompt_path": prompt_path,
        "output_path": output_path,
        "json_struct_path": json_struct_path,
        "model_name": model_name,
        "text_col": text_col,
        "id_col": id_col,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        "max_output_len": max_output_len,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": gpu_memory_utilization,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
        "checkpoint_interval": checkpoint_interval,
        "resume_from_checkpoint": resume_from_checkpoint,
        "concurrency": concurrency,
    }
    if resume_from_checkpoint:
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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    system_instr_token_length = tokenizer(system_instr, add_special_tokens=False, return_length=True)["length"]
    token_lengths, bs = [], int(1e5)
    for i in range(0, len(df), bs):
        token_lengths += tokenizer(df[text_col].iloc[i:i+bs].tolist(), add_special_tokens=False, return_length=True)["length"]
    df["token_length"] = token_lengths
    mask = system_instr_token_length + df["token_length"] > max_model_len - max_output_len
    if mask.any():
        df, df_ignored = df[~mask], df[mask]
        logger.info(f"Ignoring {mask.sum()} ({mask.mean()*100:.3f}%) samples that exceed {max_model_len-max_output_len} tokens")
        df_ignored.to_parquet(f"{output_path}/unprocessed.parquet", index=False, compression="gzip")
    # Tokenizer uses a lot of memory, free it up
    del tokenizer
    gc.collect()

    # Configure vLLM engine for Ray Data
    config = vLLMEngineProcessorConfig(
        model_source=model_path,
        engine_kwargs={
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_prefix_caching": True,  # For shared system prompt optimization
            "enable_chunked_prefill": True,  # Ray Data optimization
        },
        concurrency=concurrency,  # Number of parallel vLLM replicas
        batch_size=max_num_seqs,  # Prompts per batch
    )

    # Define preprocessing function
    def preprocess_row(row):
        """Format each row into chat messages and sampling parameters."""
        messages = [
            {"role": "system", "content": system_instr},
            {"role": "user", "content": row[text_col]},
        ]
        sampling_params = {"temperature": temperature, "max_tokens": max_output_len}
        if json_schema is not None:
            sampling_params["structured_outputs"] = StructuredOutputsParams(json=json_schema)
        return {
            "messages": messages,
            "sampling_params": sampling_params,
            id_col: row[id_col],
        }

    # Define postprocessing function
    def postprocess_row(row):
        """Extract generated text and preserve metadata."""
        return {
            id_col: row[id_col],
            "messages": row["messages"],
            "generated_output": row["generated_text"],
        }

    # Build vLLM processor with Ray Data
    logger.info("Building vLLM processor with Ray Data framework")
    vllm_processor = build_llm_processor(
        config,
        preprocess=preprocess_row,
        postprocess=postprocess_row,
        builder_kwargs={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )

    # Execute inference using Ray Data in batches
    logger.info("Starting batch inference")
    batch_size = max_num_seqs * checkpoint_interval
    for batch_num, i in enumerate(range(0, len(df), batch_size)):
        # Check if already processed
        batch_filepath = f"{output_path}/generated_output/batch_{batch_num}.parquet"
        if resume_from_checkpoint and os.path.exists(batch_filepath):
            logger.info(f"Skipping batch {batch_num}")
            continue

        # Process them
        subset = df.iloc[i:i+batch_size]
        res = vllm_processor(ray.data.from_pandas(subset))

        # Save results
        res.write_parquet(batch_filepath, compression="zstd")

    logger.info("Batch inference completed successfully")
