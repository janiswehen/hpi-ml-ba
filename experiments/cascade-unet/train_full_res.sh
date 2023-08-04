#!/bin/bash -eux

#SBATCH --job-name=cascade-unet-train-full-res
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=janis.wehen@mattermost
#SBATCH --partition=gpupro
#SBATCH --cpus-per-task=5
#SBATCH --mem=64gb
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --output=logs/cascade-unet-train-full-res-%j.log # %j is job id

user="janis.wehen"
condaenv="ba"
script_path="/dhc/home/janis.wehen/ba/models/bachelor-project/experiments/cascade-unet/train_full_res.py"
log_path="/dhc/home/janis.wehen/ba/models/bachelor-project/experiments/cascade-unet/logs/training_run_full_res.log"

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