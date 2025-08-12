import pandas as pd
import numpy as np
from datetime import datetime

# Load the data
df = pd.read_csv(r'c:\Users\tmold\AppData\Roaming\MobaXterm\slash\RemoteFiles\68052_2_0\sweep_results.csv')

# Extract hyperparameters from run_name
def extract_params(run_name):
    parts = run_name.split('_')
    lr = float(parts[0].replace('lr', '').replace('e-', 'e-'))
    dropout = float(parts[1].replace('dr', ''))
    weight_decay = float(parts[2].replace('wd', '').replace('e-', 'e-'))
    focal = 'focal' in run_name
    return lr, dropout, weight_decay, focal

# Add extracted parameters
params = df['run_name'].apply(extract_params)
df[['lr_extracted', 'dropout_extracted', 'weight_decay_extracted', 'focal_extracted']] = pd.DataFrame(params.tolist(), index=df.index)

# Get best results by validation accuracy for each configuration
best_results = df.loc[df.groupby('run_name')['val_acc'].idxmax()]
best_results = best_results.sort_values('val_acc', ascending=False)

print("=" * 80)
print("HYPERPARAMETER SWEEP IMPROVEMENT RECOMMENDATIONS")
print("=" * 80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total Experiments: {len(df)}")
print(f"Unique Configurations: {df['run_name'].nunique()}")
print(f"Best Validation Accuracy: {best_results['val_acc'].max():.2f}%")
print("=" * 80)

# 1. CURRENT PERFORMANCE ANALYSIS
print("\n1. CURRENT PERFORMANCE ANALYSIS")
print("-" * 40)
print(f"Best Configuration: {best_results.iloc[0]['run_name']}")
print(f"Best Validation Accuracy: {best_results.iloc[0]['val_acc']:.2f}%")
print(f"Best Training Accuracy: {best_results.iloc[0]['train_acc']:.2f}%")
print(f"Overfitting Gap: {best_results.iloc[0]['overfitting_gap']:.2f}%")
print(f"Average Validation Accuracy: {best_results['val_acc'].mean():.2f}%")
print(f"Standard Deviation: {best_results['val_acc'].std():.2f}%")

# 2. HYPERPARAMETER OPTIMIZATION RECOMMENDATIONS
print("\n2. HYPERPARAMETER OPTIMIZATION RECOMMENDATIONS")
print("-" * 40)

# Learning Rate Analysis
lr_analysis = best_results.groupby('lr').agg({
    'val_acc': ['mean', 'max', 'std'],
    'overfitting_gap': 'mean'
}).round(3)

best_lr = lr_analysis[('val_acc', 'max')].idxmax()
print(f"Learning Rate:")
print(f"  • Best performing: {best_lr}")
print(f"  • Recommended range: [{best_lr/2:.1e}, {best_lr*2:.1e}]")
print(f"  • Consider learning rate scheduling with initial LR = {best_lr*2:.1e}")

# Dropout Analysis
dropout_analysis = best_results.groupby('dropout').agg({
    'val_acc': ['mean', 'max', 'std'],
    'overfitting_gap': 'mean'
}).round(3)

best_dropout = dropout_analysis[('val_acc', 'max')].idxmax()
print(f"\nDropout:")
print(f"  • Best performing: {best_dropout}")
print(f"  • Recommended range: [{max(0.1, best_dropout-0.1):.1f}, {min(0.9, best_dropout+0.1):.1f}]")
print(f"  • High dropout ({best_dropout*100}%) suggests strong regularization needed")

# Weight Decay Analysis
wd_analysis = best_results.groupby('weight_decay').agg({
    'val_acc': ['mean', 'max', 'std'],
    'overfitting_gap': 'mean'
}).round(3)

best_wd = wd_analysis[('val_acc', 'max')].idxmax()
print(f"\nWeight Decay:")
print(f"  • Best performing: {best_wd}")
print(f"  • Recommended range: [{best_wd/2:.1e}, {best_wd*2:.1e}]")
print(f"  • High weight decay suggests model benefits from strong regularization")

# Focal Loss Analysis
focal_analysis = best_results.groupby('use_focal_loss').agg({
    'val_acc': ['mean', 'max', 'std'],
    'overfitting_gap': 'mean'
}).round(3)

print(f"\nLoss Function:")
if focal_analysis.loc[1, ('val_acc', 'max')] > focal_analysis.loc[0, ('val_acc', 'max')]:
    print(f"  • Focal loss performs better (max: {focal_analysis.loc[1, ('val_acc', 'max')]:.2f}% vs {focal_analysis.loc[0, ('val_acc', 'max')]:.2f}%)")
    print(f"  • Consider using focal loss for imbalanced datasets")
else:
    print(f"  • Standard loss performs better (max: {focal_analysis.loc[0, ('val_acc', 'max')]:.2f}% vs {focal_analysis.loc[1, ('val_acc', 'max')]:.2f}%)")
    print(f"  • Focal loss may not be necessary for this dataset")

# 3. OVERFITTING ANALYSIS
print("\n3. OVERFITTING ANALYSIS")
print("-" * 40)
avg_overfitting = best_results['overfitting_gap'].mean()
print(f"Average overfitting gap: {avg_overfitting:.2f}%")

if avg_overfitting > 5:
    print("  • HIGH OVERFITTING DETECTED")
    print("  • Recommendations:")
    print("    - Increase dropout rate")
    print("    - Increase weight decay")
    print("    - Reduce model complexity")
    print("    - Implement early stopping")
elif avg_overfitting < -3:
    print("  • UNDERFITTING DETECTED")
    print("  • Recommendations:")
    print("    - Decrease dropout rate")
    print("    - Decrease weight decay")
    print("    - Increase model complexity")
    print("    - Train for more epochs")
else:
    print("  • Good balance between training and validation performance")

# 4. CONVERGENCE ANALYSIS
print("\n4. CONVERGENCE ANALYSIS")
print("-" * 40)
convergence_data = df.groupby('run_name').agg({
    'epoch': 'max',
    'val_acc': ['max', 'last'],
    'train_acc': ['max', 'last']
}).round(3)

convergence_data.columns = ['max_epochs', 'best_val_acc', 'final_val_acc', 'best_train_acc', 'final_train_acc']
convergence_data['convergence_quality'] = convergence_data['best_val_acc'] - convergence_data['final_val_acc']

avg_epochs = convergence_data['max_epochs'].mean()
avg_convergence_quality = convergence_data['convergence_quality'].mean()

print(f"Average training epochs: {avg_epochs:.1f}")
print(f"Average convergence quality: {avg_convergence_quality:.2f}%")

if avg_epochs < 50:
    print("  • Models may need more training time")
    print("  • Recommendations:")
    print("    - Increase max epochs to 100-150")
    print("    - Implement learning rate scheduling")
    print("    - Use patience-based early stopping")

if avg_convergence_quality > 2:
    print("  • Poor convergence detected")
    print("  • Recommendations:")
    print("    - Implement learning rate scheduling")
    print("    - Use adaptive optimizers (Adam, AdamW)")
    print("    - Increase patience for early stopping")

# 5. SPECIFIC IMPROVEMENT STRATEGIES
print("\n5. SPECIFIC IMPROVEMENT STRATEGIES")
print("-" * 40)

print("A. IMMEDIATE IMPROVEMENTS (High Priority):")
print("   1. Use optimal hyperparameters:")
print(f"      - Learning rate: {best_lr}")
print(f"      - Dropout: {best_dropout}")
print(f"      - Weight decay: {best_wd}")
print(f"      - Loss function: {'Focal' if focal_analysis.loc[1, ('val_acc', 'max')] > focal_analysis.loc[0, ('val_acc', 'max')] else 'Standard'}")

print("\n   2. Implement learning rate scheduling:")
print("      - Start with 2x optimal LR")
print("      - Use cosine annealing or step decay")
print("      - Monitor validation loss for scheduling")

print("\n   3. Optimize training duration:")
print("      - Increase max epochs to 100-150")
print("      - Use early stopping with patience=15-20")
print("      - Monitor for convergence")

print("\nB. ADVANCED IMPROVEMENTS (Medium Priority):")
print("   1. Data augmentation:")
print("      - Implement rotation, scaling, noise addition")
print("      - Test mixup or cutmix techniques")
print("      - Consider domain-specific augmentations")

print("\n   2. Model architecture improvements:")
print("      - Test different backbone architectures")
print("      - Experiment with attention mechanisms")
print("      - Consider ensemble methods")

print("\n   3. Advanced regularization:")
print("      - Test label smoothing")
print("      - Implement stochastic depth")
print("      - Consider mixup training")

print("\nC. EXPERIMENTAL IMPROVEMENTS (Low Priority):")
print("   1. Advanced optimizers:")
print("      - Test AdamW, RAdam, or AdaBelief")
print("      - Experiment with different momentum values")
print("      - Try gradient clipping")

print("\n   2. Loss function experiments:")
print("      - Test different focal loss parameters")
print("      - Consider label smoothing")
print("      - Experiment with contrastive learning")

# 6. RECOMMENDED EXPERIMENTS
print("\n6. RECOMMENDED EXPERIMENTS")
print("-" * 40)

print("Priority 1 - Hyperparameter Fine-tuning:")
print(f"  • Test learning rates: [{best_lr/2:.1e}, {best_lr}, {best_lr*2:.1e}]")
print(f"  • Test dropout rates: [{max(0.1, best_dropout-0.1):.1f}, {best_dropout}, {min(0.9, best_dropout+0.1):.1f}]")
print(f"  • Test weight decay: [{best_wd/2:.1e}, {best_wd}, {best_wd*2:.1e}]")

print("\nPriority 2 - Training Strategy:")
print("  • Implement learning rate scheduling")
print("  • Test different early stopping strategies")
print("  • Experiment with longer training (100-150 epochs)")

print("\nPriority 3 - Advanced Techniques:")
print("  • Data augmentation pipeline")
print("  • Model ensemble methods")
print("  • Advanced regularization techniques")

# 7. EXPECTED IMPROVEMENTS
print("\n7. EXPECTED IMPROVEMENTS")
print("-" * 40)

current_best = best_results.iloc[0]['val_acc']
print(f"Current best validation accuracy: {current_best:.2f}%")

print("\nExpected improvements with recommended changes:")
print("  • Hyperparameter fine-tuning: +1-2%")
print("  • Learning rate scheduling: +0.5-1%")
print("  • Data augmentation: +1-3%")
print("  • Model ensemble: +1-2%")
print("  • Advanced regularization: +0.5-1%")

total_expected = current_best + 4  # Conservative estimate
print(f"\nConservative target: {total_expected:.2f}%")
print(f"Optimistic target: {current_best + 6:.2f}%")

# 8. IMPLEMENTATION PLAN
print("\n8. IMPLEMENTATION PLAN")
print("-" * 40)

print("Week 1:")
print("  • Implement optimal hyperparameters")
print("  • Add learning rate scheduling")
print("  • Extend training to 100-150 epochs")

print("\nWeek 2:")
print("  • Implement data augmentation")
print("  • Test different early stopping strategies")
print("  • Fine-tune hyperparameters around optimal values")

print("\nWeek 3:")
print("  • Implement model ensemble")
print("  • Test advanced regularization techniques")
print("  • Optimize training pipeline")

print("\nWeek 4:")
print("  • Final evaluation and comparison")
print("  • Documentation and reporting")
print("  • Production deployment preparation")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"• Current best: {current_best:.2f}%")
print(f"• Target improvement: +4-6%")
print(f"• Key focus areas: Hyperparameter optimization, LR scheduling, data augmentation")
print(f"• Expected timeline: 4 weeks for full implementation")
print("=" * 80)

# Save recommendations to file
with open('improvement_recommendations.txt', 'w') as f:
    f.write("HYPERPARAMETER SWEEP IMPROVEMENT RECOMMENDATIONS\n")
    f.write("=" * 80 + "\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Current Best Validation Accuracy: {current_best:.2f}%\n")
    f.write(f"Target Improvement: +4-6%\n\n")
    
    f.write("KEY RECOMMENDATIONS:\n")
    f.write(f"1. Learning Rate: {best_lr}\n")
    f.write(f"2. Dropout: {best_dropout}\n")
    f.write(f"3. Weight Decay: {best_wd}\n")
    f.write(f"4. Implement Learning Rate Scheduling\n")
    f.write(f"5. Add Data Augmentation\n")
    f.write(f"6. Extend Training to 100-150 epochs\n")

print(f"\nDetailed recommendations saved to 'improvement_recommendations.txt'")
