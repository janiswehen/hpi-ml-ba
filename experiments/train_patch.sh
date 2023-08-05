#!/bin/bash -eux

#SBATCH --job-name=task1-patch-unet-train
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=janis.wehen@mattermost
#SBATCH --partition=gpupro
#SBATCH --cpus-per-task=5
#SBATCH --mem=64gb
#SBATCH --gpus=1
#SBATCH --time=30:00:00
#SBATCH --output=logs/patch-unet-train_%j.log # %j is job id

user="janis.wehen"
condaenv="ba"
script_path="/dhc/home/janis.wehen/ba/models/bachelor-project/experiments"
log_path="/dhc/home/janis.wehen/ba/models/bachelor-project/experiments/logs/patch-unet-train.log"

function source_conda {
    eval "$(conda shell.bash hook)"
    source /dhc/home/$user/conda3/bin/activate
    conda activate $condaenv
}

function train {
    cd $script_path
    python -m unet.train_model --config configs/patch_base_config.yaml configs/ds/task01.yaml > $log_path 2>&1
}

source_conda
train