# ML4O Batch Inference

A high-performance offline batch inference pipeline using vLLM for running large language models on HPC clusters.

## Overview

This project provides an efficient solution for running batch inference workloads using vLLM on Slurm-managed computing clusters. Unlike online serving systems, this pipeline is optimized for processing large datasets offline with maximum throughput.

**Key Features:**
- Optimized for batch processing with vLLM
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

## Usage

### Basic Batch Inference

Run batch inference using the Apptainer container:

```bash
apptainer exec --nv ml4o-batch-inf-image.sif python batch_inference.py \
    --model-path /path/to/model \
    --input-path /path/to/input \
    --output-path /path/to/output \
    --batch-size 32
```

## Configuration

### Performance Tuning

Key parameters for optimization:

- `--batch-size`: Number of requests to process in parallel (default: 32)
- `--max-tokens`: Maximum tokens to generate per request (default: 512)
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (default: 1)
- `--gpu-memory-utilization`: Fraction of GPU memory to use (default: 0.9)

Example with performance tuning:

```bash
python batch_inference.py \
    --model-path /path/to/model \
    --input-file input.jsonl \
    --output-file output.jsonl \
    --batch-size 64 \
    --max-tokens 256 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95
```

## Project Structure

```
ml4o-batch-inference/
├── Dockerfile              # Container definition
├── pyproject.toml          # Project dependencies
├── README.md               # This file
├── batch_inference.py      # Main batch inference script
├── ml4o_batch_inf/         # Core package
└── examples/               # Example scripts and data
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
