#!/bin/bash
#SBATCH -p ss.gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=synclass_comparison
#SBATCH --output=synclass_comparison_%j.out

# Enhanced SynClass training script with flexible classifier selection
# Usage examples:
#   ./run_synclass.sh resnet                    # Run ResNet only
#   ./run_synclass.sh advanced fast             # Run advanced and fast classifiers
#   ./run_synclass.sh all                       # Run all available classifiers
#   ./run_synclass.sh resnet --epochs 50        # Run ResNet with custom epochs
#   ./run_synclass.sh d resnet                  # Delete previous jobs then run ResNet

# Load modules or activate conda environment if needed
# module load anaconda
# source activate myenv

# Create logs directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="result_logs/run_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=== SynClass Training Run - $TIMESTAMP ==="
echo "Log directory: $LOG_DIR"

# Parse arguments
DELETE_JOBS=false
CLASSIFIERS=()
EXTRA_ARGS=()
EPOCHS=""
LR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        d|delete)
            DELETE_JOBS=true
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            EXTRA_ARGS+=("--epochs" "$2")
            shift 2
            ;;
        --lr)
            LR="$2"
            EXTRA_ARGS+=("--lr" "$2")
            shift 2
            ;;
        all)
            CLASSIFIERS=("resnet" "advanced" "fast" "masked")
            shift
            ;;
        resnet|advanced|fast|masked|vgg3d)
            CLASSIFIERS+=("$1")
            shift
            ;;
        --*)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Available classifiers: resnet, advanced, fast, masked, vgg3d, all"
            echo "Additional options: --epochs N, --lr X, d (delete previous jobs)"
            exit 1
            ;;
    esac
done

# Default to resnet if no classifier specified
if [[ ${#CLASSIFIERS[@]} -eq 0 ]]; then
    CLASSIFIERS=("resnet")
    echo "No classifier specified, defaulting to ResNet"
fi

# Cancel previous jobs if requested
if [ "$DELETE_JOBS" = true ]; then
    echo "Deleting previous jobs..."
    squeue -u $USER -n synclass_comparison -h -o %i | grep -v $SLURM_JOB_ID | xargs -r scancel
    squeue -u $USER -n synclass_advanced -h -o %i | grep -v $SLURM_JOB_ID | xargs -r scancel
    squeue -u $USER -n synclass_resnet -h -o %i | grep -v $SLURM_JOB_ID | xargs -r scancel
else
    echo "Not deleting previous jobs (pass 'd' parameter to delete)"
fi

# Change to project directory and update code
cd $HOME/code/SynClass || { echo "Error: Cannot find SynClass directory"; exit 1; }
git pull origin main || echo "Warning: git pull failed, continuing with current code"

# Function to run a classifier
run_classifier() {
    local classifier=$1
    local log_file="$LOG_DIR/${classifier}_training.log"
    
    echo "Starting $classifier classifier training..."
    echo "Log file: $log_file"
    
    case $classifier in
        resnet)
            echo "Running ResNet classifier..." | tee -a "$log_file"
            python synapse_classifier_resnet.py "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$log_file"
            ;;
        advanced)
            echo "Running Advanced EfficientNet classifier..." | tee -a "$log_file"
            python synapse_classifier_advanced.py --model efficientnet "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$log_file"
            ;;
        fast)
            echo "Running Fast classifier..." | tee -a "$log_file"
            python synapse_classifier_fast.py "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$log_file"
            ;;
        masked)
            echo "Running Masked classifier..." | tee -a "$log_file"
            python synapse_classifier_masked.py "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$log_file"
            ;;
        vgg3d)
            echo "Running VGG3D classifier..." | tee -a "$log_file"
            python synapse_classifier_vgg3d.py "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$log_file"
            ;;
        *)
            echo "Unknown classifier: $classifier" | tee -a "$log_file"
            return 1
            ;;
    esac
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "$classifier training completed successfully" | tee -a "$log_file"
    else
        echo "$classifier training failed with exit code $exit_code" | tee -a "$log_file"
    fi
    
    return $exit_code
}

# Print run configuration
echo "=== Training Configuration ===" | tee "$LOG_DIR/run_summary.log"
echo "Classifiers to run: ${CLASSIFIERS[*]}" | tee -a "$LOG_DIR/run_summary.log"
echo "Extra arguments: ${EXTRA_ARGS[*]}" | tee -a "$LOG_DIR/run_summary.log"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_DIR/run_summary.log"
echo "================================" | tee -a "$LOG_DIR/run_summary.log"

# Check if we should run classifiers in parallel or sequentially
if [[ ${#CLASSIFIERS[@]} -eq 1 ]]; then
    # Single classifier - run directly
    echo "Running single classifier: ${CLASSIFIERS[0]}"
    run_classifier "${CLASSIFIERS[0]}"
    exit_code=$?
else
    # Multiple classifiers - check if we have enough GPU memory to run in parallel
    if command -v nvidia-smi &> /dev/null; then
        # Get GPU memory info
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
        echo "Detected GPU memory: ${GPU_MEMORY}MB"
        
        # If we have >16GB, we might be able to run 2 in parallel
        # But for safety, let's run sequentially for now
        echo "Running classifiers sequentially for stability..."
        PARALLEL=false
    else
        echo "No GPU detected, running sequentially..."
        PARALLEL=false
    fi
    
    if [ "$PARALLEL" = true ]; then
        # Run in parallel (experimental)
        echo "Running ${#CLASSIFIERS[@]} classifiers in parallel..."
        pids=()
        for classifier in "${CLASSIFIERS[@]}"; do
            run_classifier "$classifier" &
            pids+=($!)
        done
        
        # Wait for all to complete
        for pid in "${pids[@]}"; do
            wait $pid
        done
    else
        # Run sequentially (safer)
        echo "Running ${#CLASSIFIERS[@]} classifiers sequentially..."
        for classifier in "${CLASSIFIERS[@]}"; do
            echo "Starting $classifier..."
            run_classifier "$classifier"
            
            echo "Completed $classifier, waiting 10 seconds before next..."
            sleep 10
        done
    fi
fi

# Generate comparison summary
echo "=== Training Run Summary ===" | tee -a "$LOG_DIR/run_summary.log"
echo "Completed at: $(date)" | tee -a "$LOG_DIR/run_summary.log"

# Check for saved models and best accuracies from logs
for classifier in "${CLASSIFIERS[@]}"; do
    log_file="$LOG_DIR/${classifier}_training.log"
    if [[ -f "$log_file" ]]; then
        echo "--- $classifier Results ---" | tee -a "$LOG_DIR/run_summary.log"
        
        # Extract best accuracy if available
        best_acc=$(grep -o "Best val acc: [0-9.]*%" "$log_file" | tail -1 || echo "Not found")
        echo "Best validation accuracy: $best_acc" | tee -a "$LOG_DIR/run_summary.log"
        
        # Check if model was saved
        model_file=""
        case $classifier in
            resnet) model_file="best_synapse_model_resnet.pth" ;;
            advanced) model_file="best_synapse_model.pth" ;;
            fast) model_file="best_synapse_model_fast.pth" ;;
            masked) model_file="best_synapse_model_masked.pth" ;;
            vgg3d) model_file="best_synapse_model_vgg3d.pth" ;;
        esac
        
        if [[ -f "$model_file" ]]; then
            echo "Model saved: $model_file" | tee -a "$LOG_DIR/run_summary.log"
        else
            echo "Model file not found: $model_file" | tee -a "$LOG_DIR/run_summary.log"
        fi
        echo "" | tee -a "$LOG_DIR/run_summary.log"
    fi
done

echo "All training runs completed!"
echo "Check logs in: $LOG_DIR"
echo "Summary: $LOG_DIR/run_summary.log"