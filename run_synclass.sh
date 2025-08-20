#!/bin/bash

# Launcher script for submitting multiple independent SLURM jobs for a hyperparameter sweep.
# Usage: ./run_synclass.sh
# --> EDIT THE PARTITION VARIABLE أدناه TO SWITCH BETWEEN CPU AND GPU <--

echo "--- Starting 2D CNN Multi-Channel Hyperparameter Sweep ---"

# --- Update codebase ---
echo "Pulling latest changes from git..."
git pull
echo "---"

# --- Architecture hyperparameters for 2D CNN Multi-Channel sweep ---
CNN_DEPTHS=(1 2 3)                      # Different CNN depths (1, 2, or 3 conv blocks)
LEARNING_RATE=1e-5                      # Fixed learning rate
DROPOUT_RATES=(0.3 0.5 0.7)             # Test different dropout rates for overfitting control

# --- SLURM Configuration ---
PARTITION="ss.gpu" # Set to "ss.gpu" to automatically request a GPU
TIME="48:00:00"
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
MASTER_SWEEP_DIR="sweep_${SWEEP_TIMESTAMP}"
mkdir -p "$MASTER_SWEEP_DIR"
echo "Master sweep directory: $MASTER_SWEEP_DIR"

for CNN_DEPTH in "${CNN_DEPTHS[@]}"; do
    for DROPOUT in "${DROPOUT_RATES[@]}"; do
      
        # Define unique names for the job and its output files
        RUN_NAME="2dcnn_mc_d${CNN_DEPTH}_dr${DROPOUT}"
        
        JOB_NAME="synclass_${RUN_NAME}"
        OUTPUT_LOG="${MASTER_SWEEP_DIR}/${RUN_NAME}.out"
        
        # Construct the python command to be executed by SLURM
        # Note: The path to the script is now relative, assuming submission from project root
        PYTHON_CMD="python -W ignore synapse_classifier_2dcnn_multichannel.py \
            --cnn_depth ${CNN_DEPTH} \
            --lr ${LEARNING_RATE} \
            --dropout_rate ${DROPOUT} \
            --epochs 100 \
            --batch_size 64 \
            --input_size 224 \
            --run_name ${RUN_NAME}"

        # Use sbatch with --wrap to submit the command as a job
        echo "Submitting job: $JOB_NAME"
      
        # Prepare the full command for --wrap
        FULL_CMD="cd $HOME/code/SynClass && SWEEP_MASTER_DIR=$MASTER_SWEEP_DIR $PYTHON_CMD"
        
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

echo "--- All hyperparameter jobs have been submitted. ---"
echo ""
echo "To analyze results after completion, run:"
echo "python analyze_sweep_results.py $MASTER_SWEEP_DIR"
echo ""
echo "Or wait for all jobs to complete and then run:"
echo "python analyze_sweep_results.py $MASTER_SWEEP_DIR"