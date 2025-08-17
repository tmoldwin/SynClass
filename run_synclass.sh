#!/bin/bash

# Launcher script for submitting multiple independent SLURM jobs for a hyperparameter sweep.
# Usage: ./run_synclass.sh
# --> EDIT THE PARTITION VARIABLE أدناه TO SWITCH BETWEEN CPU AND GPU <--

echo "--- Starting Hyperparameter Sweep Job Submission ---"

# --- Update codebase ---
echo "Pulling latest changes from git..."
git pull
echo "---"

# --- Architecture hyperparameters for ResNet sweep ---
RESNET_DEPTHS=(50 101 152)            # 3 deepest ResNet variants
CLASSIFIER_WIDTHS=(64 128 256)        # 3 widest classifier widths

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
MASTER_SWEEP_DIR="sweep_${SWEEP_TIMESTAMP}"
mkdir -p "$MASTER_SWEEP_DIR"
echo "Master sweep directory: $MASTER_SWEEP_DIR"

for RESNET_DEPTH in "${RESNET_DEPTHS[@]}"; do
    for CLASSIFIER_WIDTH in "${CLASSIFIER_WIDTHS[@]}"; do
      
    # Define unique names for the job and its output files
    RUN_NAME="resnet_d${RESNET_DEPTH}_w${CLASSIFIER_WIDTH}"
        
        JOB_NAME="synclass_${RUN_NAME}"
        OUTPUT_LOG="${MASTER_SWEEP_DIR}/${RUN_NAME}.out"
        
        # Construct the python command to be executed by SLURM
        # Note: The path to the script is now relative, assuming submission from project root
                  PYTHON_CMD="python -W ignore synapse_classifier_resnet.py \
                      --resnet_depth ${RESNET_DEPTH} \
            --classifier_width ${CLASSIFIER_WIDTH} \
          --epochs 150 \
          --run_name ${RUN_NAME}"

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

echo "--- All hyperparameter jobs have been submitted. ---"
echo ""
echo "To analyze results after completion, run:"
echo "python analyze_sweep_results.py $MASTER_SWEEP_DIR"
echo ""
echo "Or wait for all jobs to complete and then run:"
echo "python analyze_sweep_results.py $MASTER_SWEEP_DIR"