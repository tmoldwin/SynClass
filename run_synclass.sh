#!/bin/bash
#SBATCH -p gpu.q
#SBATCH --gres=gpu:1
#SBATCH --job-name=synclass
#SBATCH --output=synclass_%j.out

# Load modules or activate conda environment if needed
# module load anaconda
# source activate myenv

cd $HOME/code/SynClass
git pull origin main
python synapse_classifier_resnet.py