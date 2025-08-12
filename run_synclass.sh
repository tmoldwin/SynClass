#!/bin/bash

# Launcher script for submitting multiple independent SLURM jobs for a hyperparameter sweep.
# Usage: ./run_synclass.sh
# --> EDIT THE PARTITION VARIABLE أدناه TO SWITCH BETWEEN CPU AND GPU <--

echo "--- Starting Hyperparameter Sweep Job Submission ---"

# --- Update codebase ---
echo "Pulling latest changes from git..."
git pull
echo "---"

# --- IMPROVED Hyperparameter Grid (based on analysis) ---
LEARNING_RATES=(5e-6 2e-6 1e-6)  # Keep optimal LR from sweep
DROPOUT_RATES=(0.3 0.4 0.5)      # REDUCED from 0.6-0.8 (less over-regularization)
WEIGHT_DECAYS=(5e-4 1e-3 2e-3)   # Keep same range
USE_FOCAL_LOSS=(true false)      # Test both loss functions

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
SWEEP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SWEEP_LOG_DIR="result_logs/sweep_${SWEEP_TIMESTAMP}"
MASTER_SWEEP_DIR="sweep_${SWEEP_TIMESTAMP}"
mkdir -p "$SWEEP_LOG_DIR"
mkdir -p "$MASTER_SWEEP_DIR"
echo "Master sweep directory: $MASTER_SWEEP_DIR"
echo "Log directory for this sweep: $SWEEP_LOG_DIR"

for LR in "${LEARNING_RATES[@]}"; do
  for DROPOUT in "${DROPOUT_RATES[@]}"; do
    for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
      for FOCAL_LOSS in "${USE_FOCAL_LOSS[@]}"; do
      
        # Define unique names for the job and its output files
        RUN_NAME="lr${LR}_dr${DROPOUT}_wd${WEIGHT_DECAY}"
        if [ "$FOCAL_LOSS" = true ]; then
          RUN_NAME+="_focal"
        fi
        
        JOB_NAME="synclass_${RUN_NAME}"
        OUTPUT_LOG="${SWEEP_LOG_DIR}/${RUN_NAME}.out"
        
        # Construct the python command to be executed by SLURM
        # Note: The path to the script is now relative, assuming submission from project root
        PYTHON_CMD="python synapse_classifier_resnet.py \
          --lr ${LR} \
          --dropout_rate ${DROPOUT} \
          --weight_decay ${WEIGHT_DECAY} \
          --epochs 150 \
          --run_name ${RUN_NAME}"
          
        if [ "$FOCAL_LOSS" = true ]; then
          PYTHON_CMD+=" --use_focal_loss"
        fi

        # Use sbatch with --wrap to submit the command as a job
        echo "Submitting job: $JOB_NAME"
      
        # Prepare the full command for --wrap
        FULL_CMD="cd $HOME/code/SynClass && pip install -r requirements.txt && SWEEP_MASTER_DIR=$MASTER_SWEEP_DIR $PYTHON_CMD"
        
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
done

echo "--- All hyperparameter jobs have been submitted. ---"
echo ""
echo "To analyze results after completion, run:"
echo "python analyze_sweep_results.py $MASTER_SWEEP_DIR"
echo ""
echo "Or wait for all jobs to complete and then run:"
echo "python analyze_sweep_results.py $MASTER_SWEEP_DIR"