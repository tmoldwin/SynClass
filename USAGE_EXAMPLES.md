# SynClass Training Script Usage Examples

The enhanced `run_synclass.sh` script allows flexible training of different synapse classifiers with various parameters.

## Basic Usage

### Run Single Classifier
```bash
# Run ResNet classifier (default if no classifier specified)
./run_synclass.sh

# Run specific classifier
./run_synclass.sh resnet
./run_synclass.sh advanced
./run_synclass.sh fast
./run_synclass.sh masked
./run_synclass.sh vgg3d
```

### Run Multiple Classifiers
```bash
# Run multiple classifiers sequentially
./run_synclass.sh resnet advanced fast

# Run all available classifiers
./run_synclass.sh all
```

## Advanced Usage

### Custom Training Parameters
```bash
# Run ResNet with custom epochs
./run_synclass.sh resnet --epochs 50

# Run with custom learning rate
./run_synclass.sh advanced --lr 1e-4

# Run with both custom epochs and learning rate
./run_synclass.sh resnet --epochs 100 --lr 2e-6

# Run multiple classifiers with same parameters
./run_synclass.sh resnet advanced --epochs 75 --lr 5e-5
```

### Job Management
```bash
# Delete previous jobs before starting new ones
./run_synclass.sh d resnet
./run_synclass.sh delete advanced fast

# Alternative syntax
./run_synclass.sh resnet advanced d
```

## Output and Logging

### Log Organization
- All runs create timestamped log directories in `result_logs/run_YYYYMMDD_HHMMSS/`
- Individual classifier logs: `{classifier}_training.log`
- Summary of all runs: `run_summary.log`

### Output Files
```
result_logs/run_20241201_143022/
├── run_summary.log           # Overall summary and comparison
├── resnet_training.log       # ResNet training output
├── advanced_training.log     # Advanced classifier output
└── fast_training.log         # Fast classifier output
```

## Comparison Examples

### Quick Comparison
```bash
# Compare ResNet vs Advanced classifier
./run_synclass.sh resnet advanced --epochs 50

# Compare all classifiers with shorter training
./run_synclass.sh all --epochs 25
```

### Full Comparison with Different Parameters
```bash
# Run comprehensive comparison
./run_synclass.sh all --epochs 100 --lr 1e-5
```

## Slurm Integration

The script is designed to work with Slurm job scheduling:

```bash
# Submit as Slurm job
sbatch run_synclass.sh resnet advanced

# Submit with specific GPU requirements
sbatch --gres=gpu:2 run_synclass.sh all
```

## Tips

1. **Sequential vs Parallel**: Currently runs classifiers sequentially for stability
2. **GPU Memory**: Script automatically detects GPU memory and adjusts batch sizes
3. **Log Files**: All output is saved with timestamps for easy comparison
4. **Model Files**: Each classifier saves its best model with a unique name
5. **Summary**: Check `run_summary.log` for quick comparison of results

## Example Workflow

```bash
# 1. Quick test with short training
./run_synclass.sh resnet --epochs 10

# 2. Compare top performers
./run_synclass.sh resnet advanced --epochs 50

# 3. Full training run
./run_synclass.sh all --epochs 100

# 4. Check results
cat result_logs/run_*/run_summary.log
``` 