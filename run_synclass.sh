#!/bin/bash
#SBATCH -p ss.gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=synclass_advanced
#SBATCH --output=synclass_advanced_%j.out

# Load modules or activate conda environment if needed
# module load anaconda
# source activate myenv

# Cancel all previous jobs for this specific project (synclass), excluding current job
squeue -u $USER -n synclass_advanced -h -o %i | grep -v $SLURM_JOB_ID | xargs -r scancel

cd $HOME/code/SynClass
git pull origin main || echo "Warning: git pull failed, continuing with current code"
python synapse_classifier_advanced.py --model efficientnet --epochs 200 --lr 3e-4