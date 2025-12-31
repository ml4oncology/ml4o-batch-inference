#!/bin/bash
#SBATCH --job-name=batch-inference
#SBATCH --partition=gpu
#SBATCH --account=grantgroup_gpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/cluster/home/%u/logs/%j.out
#SBATCH --error=/cluster/home/%u/logs/%j.err

mkdir -p /cluster/home/$USER/logs

# Load Apptainer module
module load apptainer

# Load the paths
# source .env
MODEL_PATH="/cluster/projects/gliugroup/2BLAST/LLMs"
IMAGE_PATH="/cluster/projects/gliugroup/2BLAST/containers/ml4o-batch-inf-image.sif"

# Validate required files exist                                    
if [ ! -f "$IMAGE_PATH" ]; then                                    
    echo "Error: Container image not found at $IMAGE_PATH"         
    exit 1                                                         
fi                                                                 
                                                                   
if [ ! -d "$MODEL_PATH" ]; then                                    
    echo "Error: Model directory not found at $MODEL_PATH"         
    exit 1                                                         
fi

# Print job information                                            
echo "================================================"            
echo "Job ID: $SLURM_JOB_ID"                                       
echo "Job Name: $SLURM_JOB_NAME"                                   
echo "Node: $SLURM_NODELIST"                                       
echo "Number of GPUs: $SLURM_GPUS"                                 
echo "Start Time: $(date)"                                         
echo "================================================"

# Set up bind paths
# NOTE: because we are not using --containall, apptainer automatically binds $HOME, /tmp, /dev, /proc, /sys, and ./
export APPTAINER_BINDPATH=$APPTAINER_BINDPATH,$MODEL_PATH

# Run batch inference script inside the container
apptainer exec --nv $IMAGE_PATH python3.10 batch_inference.py \
    --data-path example-data.parquet \
    --output-path example-output.parquet \
    --prompt-path example-prompt.txt \

# Print completion information                                     
echo "================================================"            
echo "Job completed at: $(date)"                                   
echo "================================================"