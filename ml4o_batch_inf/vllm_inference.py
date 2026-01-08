"""
Single-node inference pipeline using vLLM.

References
----------
https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/prefix_caching.py
https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/structured_outputs.py
"""
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
    tokenizer_path: str | None,
) -> None:
    """Run batch inference using vLLM."""
    os.makedirs(f"{output_path}/generated_output", exist_ok=True)
    model_path = f"{ROOT_DIR}/LLMs/{model_name}"
    if tokenizer_path is None:
        tokenizer_path = model_path

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
        "tokenizer_path": tokenizer_path,
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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
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

    # Initialize vLLM
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,

        # pre-allocates KV cache memory based on max-model-len, regardless of actual
        # prompt length. A large max-model-len will eat up a large chunk of GPU memory,
        # leaving little memory for the actual inputs/outputs
        max_model_len=max_model_len,

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
         for text in df[text_col]
    ]

    # Create sampling parameter object
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_output_len)
    if json_schema is not None:
        # Ensure sturctured JSON output
        structured_outputs_params_json = StructuredOutputsParams(json=json_schema)
        sampling_params.structured_outputs = structured_outputs_params_json

    # Warmup so that the shared prompt's KV cache is computed (for prefix caching)
    llm.chat(prompts[0], sampling_params)

    # Process the prompts in batches
    batch_size = max_num_seqs * checkpoint_interval
    for batch_num, i in enumerate(tqdm(range(0, len(prompts), batch_size))):
        # Check if already processed
        batch_filepath = f"{output_path}/generated_output/batch_{batch_num}.parquet"
        if resume_from_checkpoint and os.path.exists(batch_filepath):
            logger.info(f"Skipping batch {batch_num}")
            continue

        # Process them
        batch = prompts[i:i+batch_size]
        ids = df[id_col].iloc[i:i+batch_size]
        outputs = llm.chat(batch, sampling_params, chat_template_kwargs={"enable_thinking": enable_thinking})
        res = [{"prompt": output.prompt, "generated_output": output.outputs[0].text} for output in outputs]

        # Save results
        res = pd.DataFrame(res, index=ids)
        res.to_parquet(batch_filepath, compression="zstd")
