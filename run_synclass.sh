#!/bin/bash
#SBATCH -p ss.gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=synclass_hparam
#SBATCH --output=synclass_hparam_%A_%a.out
#SBATCH --array=0-26

# Enhanced SynClass training script for hyperparameter sweep
# Usage: ./run_synclass.sh

# --- Hyperparameter Grid ---
LEARNING_RATES=(1e-5 5e-6 2e-6)
DROPOUT_RATES=(0.3 0.5 0.7)
WEIGHT_DECAYS=(1e-4 5e-4 1e-3)

# Calculate total number of jobs
NUM_LRS=${#LEARNING_RATES[@]}
NUM_DROPOUTS=${#DROPOUT_RATES[@]}
NUM_DECAYS=${#WEIGHT_DECAYS[@]}

# Map SLURM_ARRAY_TASK_ID to hyperparameter combination
LR_I=$((SLURM_ARRAY_TASK_ID / (NUM_DROPOUTS * NUM_DECAYS) % NUM_LRS))
DR_I=$((SLURM_ARRAY_TASK_ID / NUM_DECAYS % NUM_DROPOUTS))
WD_I=$((SLURM_ARRAY_TASK_ID % NUM_DECAYS))

LR=${LEARNING_RATES[$LR_I]}
DROPOUT=${DROPOUT_RATES[$DR_I]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$WD_I]}

# Create logs directory with timestamp and job ID
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="task_${SLURM_ARRAY_TASK_ID}_lr${LR}_dr${DROPOUT}_wd${WEIGHT_DECAY}"
LOG_DIR="result_logs/hparam_sweep_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

echo "=== SynClass Hyperparameter Sweep ===" > "$LOG_FILE"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}, Task ID: ${SLURM_ARRAY_TASK_ID}" >> "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" >> "$LOG_FILE"
echo "--- Parameters ---" >> "$LOG_FILE"
echo "Learning Rate: $LR" >> "$LOG_FILE"
echo "Dropout Rate: $DROPOUT" >> "$LOG_FILE"
echo "Weight Decay: $WEIGHT_DECAY" >> "$LOG_FILE"
echo "Log File: $LOG_FILE" >> "$LOG_FILE"
echo "==============================" >> "$LOG_FILE"

# Change to project directory and update code
cd $HOME/code/SynClass || { echo "Error: Cannot find SynClass directory"; exit 1; }
git pull origin main || echo "Warning: git pull failed, continuing with current code"

# Run the training script with the selected hyperparameters
echo "Starting ResNet classifier training with HParams..." | tee -a "$LOG_FILE"
python synapse_classifier_resnet.py \
    --lr "$LR" \
    --dropout_rate "$DROPOUT" \
    --weight_decay "$WEIGHT_DECAY" \
    --epochs 100 \
    --run_name "$RUN_NAME" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully" | tee -a "$LOG_FILE"
else
    echo "Training failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
fi

echo "Job finished at: $(date)" >> "$LOG_FILE"