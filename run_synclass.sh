#!/bin/bash

# Launcher script for submitting multiple independent SLURM jobs for a hyperparameter sweep.
# Usage: ./run_synclass.sh
# --> EDIT THE PARTITION VARIABLE أدناه TO SWITCH BETWEEN CPU AND GPU <--

echo "--- Starting Hyperparameter Sweep Job Submission ---"

# --- Update codebase ---
echo "Pulling latest changes from git..."
git pull
echo "---"

# --- Hyperparameter Grid ---
LEARNING_RATES=(1e-6 1e-7 1e-8 1e-9)
DROPOUT_RATES=(0.3 0.5 0.7)
WEIGHT_DECAYS=(1e-4 5e-4 1e-3)

# --- SLURM Configuration ---
PARTITION="ss.gpu" # Set to "ss.gpu" to automatically request a GPU
TIME="24:00:00"
GRES="" # Will be set automatically based on the partition name

# Automatically add GPU request if partition name contains "gpu"
if [[ "$PARTITION" == *"gpu"* ]]; then
  GRES="gpu:1"
  echo "GPU partition detected. Requesting GPU: $GRES"
else
  echo "CPU partition detected. No GPU requested."
fi

# --- Main Loop for Job Submission ---
SWEEP_LOG_DIR="result_logs/sweep_$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$SWEEP_LOG_DIR"
echo "Log directory for this sweep: $SWEEP_LOG_DIR"

for LR in "${LEARNING_RATES[@]}"; do
  for DROPOUT in "${DROPOUT_RATES[@]}"; do
    for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
      
      # Define unique names for the job and its output files
      RUN_NAME="lr${LR}_dr${DROPOUT}_wd${WEIGHT_DECAY}"
      JOB_NAME="synclass_${RUN_NAME}"
      OUTPUT_LOG="${SWEEP_LOG_DIR}/${RUN_NAME}.out"
      
      # Construct the python command to be executed by SLURM
      # Note: The path to the script is now relative, assuming submission from project root
      PYTHON_CMD="python synapse_classifier_resnet.py \
        --lr ${LR} \
        --dropout_rate ${DROPOUT} \
        --weight_decay ${WEIGHT_DECAY} \
        --epochs 100 \
        --run_name ${RUN_NAME}"

      # Use sbatch with --wrap to submit the command as a job
      echo "Submitting job: $JOB_NAME"
      
      # Prepare the full command for --wrap
      FULL_CMD="cd $HOME/code/SynClass && pip install -r requirements.txt && $PYTHON_CMD"
      
      SBATCH_CMD="sbatch \
        --job-name=\"$JOB_NAME\" \
        --partition=\"$PARTITION\" \
        --time=\"$TIME\" \
        --output=\"$OUTPUT_LOG\""

      if [[ -n "$GRES" ]]; then
        SBATCH_CMD="$SBATCH_CMD --gres=\"$GRES\""
      fi

      SBATCH_CMD="$SBATCH_CMD --wrap=\"$FULL_CMD\""
      
      # Execute the sbatch command
      eval $SBATCH_CMD
      
      # Small delay to avoid overwhelming the SLURM scheduler
      sleep 1
      
    done
  done
done

echo "--- All hyperparameter jobs have been submitted. ---"