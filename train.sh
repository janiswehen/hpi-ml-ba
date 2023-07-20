#!/bin/bash -eux

#SBATCH --job-name=full-unet-train
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=janis.wehen@mattermost
#SBATCH --partition=gpupro
#SBATCH --cpus-per-task=5
#SBATCH --mem=64gb
#SBATCH --gpus=1
#SBATCH --time=16:00:00
#SBATCH --output=logs/full-unet-train_%j.log # %j is job id

user="janis.wehen"
condaenv="full-unet"
script_path="/dhc/home/janis.wehen/ba/models/Full-3D-UNet/train.py"
log_path="/dhc/home/janis.wehen/ba/models/Full-3D-UNet/training_run.log"

function source_conda {
    eval "$(conda shell.bash hook)"
    source /dhc/home/$user/conda3/bin/activate
    conda activate $condaenv
}

function train {
    python $script_path > $log_path 2>&1
}

source_conda
train