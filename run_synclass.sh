#!/bin/bash
#SBATCH -p ss.gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=synclass
#SBATCH --output=synclass_%j.out

# Load modules or activate conda environment if needed
# module load anaconda
# source activate myenv

# Cancel all previous jobs for this specific project (synclass)
scancel -n synclass -u $USER

cd $HOME/code/SynClass
git pull origin main || echo "Warning: git pull failed, continuing with current code"
python synapse_classifier_resnet.py