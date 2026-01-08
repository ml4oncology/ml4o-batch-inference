"""CLI for ML4O Batch Inference."""
import click

from ml4o_batch_inf import vllm_inference, ray_inference


# Shared options decorator
def common_options(func):
    """Shared options for both vllm and ray commands."""
    options = [
        click.option("--data-path", required=True, help="Path to the parquet dataset"),
        click.option("--prompt-path", required=True, help="Path to the text file containing the system prompt"),
        click.option("--output-path", required=True, help="Directory to save the results"),
        click.option("--json-struct-path", default=None, help="Path to the JSON file describing the desired structured output"),
        click.option("--model-name", default="Qwen3-14B", help="Model name (HuggingFace-style model directory)"),
        click.option("--text-col", default="note", help="Name of the column containing the text"),
        click.option("--id-col", default="note_id", help="Name of the column containing the text identifier"),
        click.option("--tensor-parallel-size", default=1, help="Number of GPUs for tensor parallelism"),
        click.option("--max-model-len", default=4096, help="Max number of tokens for prompt + output"),
        click.option("--max-output-len", "--max-tokens", default=512, help="Max number of tokens for output"),
        click.option("--max-num-seqs", "--batch-size", default=16, help="Max number of prompts processed in parallel"),
        click.option("--gpu-memory-utilization", default=0.95, help="Fraction of GPU memory to use"),
        click.option("--temperature", default=0.8, help=">1.0: More random. 0.0: Greedy. 1.0: Standard"),
        click.option("--enable-thinking", is_flag=True, help="Enable thinking (uses more output tokens)"),
        click.option("--checkpoint-interval", default=100, help="Save checkpoint every N batches"),
        click.option("--resume-from-checkpoint", is_flag=True, help="Resume from existing output directory"),
    ]
    for option in reversed(options):
        func = option(func)
    return func


@click.group()
@click.version_option()
def cli():
    """ML4O Batch Inference - Run LLM inference on datasets."""
    pass


@cli.command()
@common_options
@click.option("--tokenizer-path", default=None, help="Path to the tokenizer (defaults to model path)")
def vllm(**kwargs):
    """Run batch inference using vLLM (single node)."""
    vllm_inference.run(**kwargs)


@cli.command()
@common_options
@click.option("--concurrency", default=1, help="Number of parallel vLLM replicas for Ray Data")
def ray(**kwargs):
    """Run distributed batch inference using vLLM + Ray Data."""
    ray_inference.run(**kwargs)


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
