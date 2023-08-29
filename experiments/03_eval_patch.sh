#!/bin/bash -eux

#SBATCH --job-name=03-eval-patch
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=janis.wehen@mattermost
#SBATCH --partition=gpupro
#SBATCH --cpus-per-task=10
#SBATCH --mem=64gb
#SBATCH --gpus=1
#SBATCH --time=30:00:00
#SBATCH --output=logs/03_eval_patch_%j.log # %j is job id

user="janis.wehen"
condaenv="ba"
script_path="/dhc/home/janis.wehen/ba/models/bachelor-project/experiments"
log_path="/dhc/home/janis.wehen/ba/models/bachelor-project/experiments/logs/03_eval_patch.log"

function source_conda {
    eval "$(conda shell.bash hook)"
    source /dhc/home/$user/conda3/bin/activate
    conda activate $condaenv
}

function train {
    cd $script_path
    python -m unet.eval_model --config configs/eval/patch_base_eval_config.yaml configs/dataset/task03.yaml > $log_path 2>&1
}

source_conda
train
