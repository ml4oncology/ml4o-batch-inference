# ML4O Batch Inference

A high-performance offline batch inference pipeline using vLLM for running large language models on HPC clusters.

## Overview

This project provides an efficient solution for running batch inference workloads using vLLM on Slurm-managed computing clusters. Unlike online serving systems, this pipeline is optimized for processing large datasets offline with maximum throughput.

**Key Features:**
- Optimized for batch processing with vLLM
- Single-node and distributed (Ray) inference modes
- CLI with `ml4o-batch-inf` command
- Apptainer/Singularity container support for HPC environments

## Installation

### Environment Setup

Use the provided [`Dockerfile`](Dockerfile) to build the Docker image **(GPU machine required)**:

```bash
docker build --progress=plain -t ml4o-batch-inf-image .
docker save -o ml4o-batch-inf-image.tar ml4o-batch-inf-image:latest
```

### Upload to H4H Cluster

- Upload the `.tar` file to the H4H Cluster
- **Important:** Before uploading, check the cluster to see if the `.tar` file already exists to avoid redundant uploads

### Convert to SIF (Apptainer)

Convert the Docker image `.tar` file to a `.sif` file using **Apptainer**:

```bash
module load apptainer  # NOTE: current version v1.4.1
export APPTAINER_CACHEDIR=/tmp  # NOTE: in case your home directory runs out of space
apptainer build ml4o-batch-inf-image.sif docker-archive://ml4o-batch-inf-image.tar
```

Alternatively, you can set up Apptainer locally, convert the files locally, and upload the `.sif` file to the H4H cluster.
```bash
docker build --progress=plain -t ml4o-batch-inf-image .
apptainer build ml4o-batch-inf-image.sif docker-daemon://ml4o-batch-inf-image
```
## Usage

### CLI Commands

```bash
# Show help
ml4o-batch-inf --help
ml4o-batch-inf vllm --help
ml4o-batch-inf ray --help
```

### Single-Node Inference (vLLM)

Run batch inference on a single node:

```bash
apptainer exec --nv ml4o-batch-inf-image.sif ml4o-batch-inf vllm \
    --data-path /path/to/data.parquet \
    --output-path /path/to/output \
    --prompt-path /path/to/prompt.txt
```

### Distributed Inference (Ray + vLLM)

Run distributed batch inference across multiple GPUs:

```bash
apptainer exec --nv ml4o-batch-inf-image.sif ml4o-batch-inf ray \
    --data-path /path/to/data.parquet \
    --output-path /path/to/output \
    --prompt-path /path/to/prompt.txt \
    --concurrency 2
```

### Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-path` | required | Path to the parquet dataset |
| `--prompt-path` | required | Path to the system prompt file |
| `--output-path` | required | Directory to save results |
| `--model-name` | `Qwen3-14B` | Model name |
| `--text-col` | `note` | Column containing text |
| `--id-col` | `note_id` | Column containing text identifier |
| `--max-model-len` | `4096` | Max tokens for prompt + output |
| `--max-output-len` | `512` | Max tokens for output |
| `--temperature` | `0.8` | Sampling temperature |
| `--tensor-parallel-size` | `1` | GPUs for tensor parallelism |
| `--enable-thinking` | `false` | Enable thinking mode |
| `--resume-from-checkpoint` | `false` | Resume from existing output |

**vLLM-only:** `--tokenizer-path`

**Ray-only:** `--concurrency` (number of parallel vLLM replicas)

## Project Structure

```
ml4o-batch-inference/
├── Dockerfile                        # Container definition
├── pyproject.toml                    # Project dependencies
├── README.md                         # This file
├── example_batch_inference.slurm     # Example Slurm script (single-node)
├── example_ray_batch_inference.slurm # Example Slurm script (distributed)
└── ml4o_batch_inf/                   # Core package
    ├── cli.py                        # CLI entry point
    ├── vllm_inference.py             # Single-node vLLM inference
    └── ray_inference.py              # Distributed Ray + vLLM inference
```

## Development

### Local Installation

For development purposes, install the package locally:

```bash
pip install -e .[dev]
```

### Running Tests

```bash
pytest tests/
```

### Code Quality

This project uses:
- `ruff` for linting and formatting
- `mypy` for type checking
- `pre-commit` for git hooks

Set up pre-commit hooks:

```bash
pre-commit install
```

## Supported Models

Models supported by vLLM can be found in the [official documentation](https://docs.vllm.ai/en/stable/models/supported_models.html).

For specific model weights locations on the H4H cluster, contact the ML4O team.

## Troubleshooting

### Out of Memory Errors

- Reduce `--batch-size`
- Reduce `--max-tokens`
- Lower `--gpu-memory-utilization`
- Use tensor parallelism across multiple GPUs

### Slow Inference

- Increase `--batch-size` (if memory allows)
- Use tensor parallelism with `--tensor-parallel-size`
- Check GPU utilization with `nvidia-smi`

### Container Issues

- Ensure `--nv` flag is used with Apptainer for GPU access
- Check CUDA compatibility between host and container
- Verify GPU allocation in Slurm job

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact the ML4O team at UHN.

## Acknowledgments

Built on top of:
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM inference engine
- [Vector Institute's vector-inference](https://github.com/VectorInstitute/vector-inference) - Inspiration for cluster deployment patterns
