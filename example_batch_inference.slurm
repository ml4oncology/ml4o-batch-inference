#!/bin/bash
#SBATCH --job-name=batch-inference
#SBATCH --partition=gpu
#SBATCH --account=grantgroup_gpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# Load Apptainer module
module load apptainer

# Load the paths
# source .env
MODEL_PATH="/cluster/projects/gliugroup/2BLAST/LLMs"
IMAGE_PATH="/cluster/projects/gliugroup/2BLAST/containers/ml4o-batch-inf-image.sif"

# Set up bind paths
# NOTE: because we are not using --containall, apptainer automatically binds $HOME, /tmp, /dev, /proc, /sys, and ./
export APPTAINER_BINDPATH=$APPTAINER_BINDPATH,$MODEL_PATH

# Run batch inference script inside the vec-inf container
apptainer exec --nv $IMAGE_PATH python3.10 batch_inference.py \
    --data-path example-data.parquet \
    --output-path example-output.parquet \
    --prompt-path example-prompt.txt \
