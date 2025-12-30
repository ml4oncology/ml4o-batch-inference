import argparse

import pandas as pd
from vllm import LLM, SamplingParams


ROOT_DIR = "/cluster/projects/gliugroup/2BLAST"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset in parquet format")
    parser.add_argument("--output-path", type=str, help="Where to save the results")
    parser.add_argument("--prompt-path", type=str, required=True, help="Path to the text file containing the system prompt")
    parser.add_argument("--text-col", type=str, default="note", help="Name of the column containing the text")
    parser.add_argument("--id-col", type=str, default="note_id", help="Name of the column containing the text identifier")
    parser.add_argument("--model-name", type=str, default="Qwen3-14B")
    args = parser.parse_args()

    # Load dataset and system instruction
    df = pd.read_parquet(args.data_path, columns=[args.id_col, args.text_col])
    with open(args.prompt_path, "r", encoding="utf-8") as file:
        system_instr = file.read()

    # Initialize with 2 GPUs
    llm = LLM(
        model=f"{ROOT_DIR}/LLMs/{args.model_name}",
        tensor_parallel_size=2,
        max_model_len=2048,
        enable_prefix_caching=True
    )

    sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

    # Create the prompts
    prompts = [(system_instr + df[args.text_col]).tolist()]

    # Process the prompts
    outputs = llm.generate(prompts, sampling_params)

    # Save results
    res = [output.outputs[0].text for output in outputs]
    df["generated_output"] = res
    df.to_parquet(args.output_path, compression="zstd", index=False)
