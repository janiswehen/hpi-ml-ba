#!/bin/bash -eux

#SBATCH --job-name=task1-full-unet-train
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=janis.wehen@mattermost
#SBATCH --partition=gpupro
#SBATCH --cpus-per-task=5
#SBATCH --mem=64gb
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --output=logs/full-unet-train_%j.log # %j is job id

user="janis.wehen"
condaenv="ba"
script_path="/dhc/home/janis.wehen/ba/models/bachelor-project/experiments"
log_path="/dhc/home/janis.wehen/ba/models/bachelor-project/experiments/logs/training_run_1.1full.log"

function source_conda {
    eval "$(conda shell.bash hook)"
    source /dhc/home/$user/conda3/bin/activate
    conda activate $condaenv
}

function train {
    cd $script_path
    python -m unet.train_model --config configs/full/full_base_config.yaml configs/full/task01_full_config.yaml > $log_path 2>&1
}

source_conda
train