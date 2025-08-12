import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv(r'c:\Users\tmold\AppData\Roaming\MobaXterm\slash\RemoteFiles\68052_2_0\sweep_results.csv')

print("=== HYPERPARAMETER SWEEP ANALYSIS ===\n")
print(f"Total experiments: {len(df)}")
print(f"Unique configurations: {df['run_name'].nunique()}")
print(f"Epochs per experiment: {df.groupby('run_name')['epoch'].max().mean():.1f}")

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

# Verify extraction matches original columns
print(f"Parameter extraction accuracy: {(df['lr'] == df['lr_extracted']).all() and (df['dropout'] == df['dropout_extracted']).all() and (df['weight_decay'] == df['weight_decay_extracted']).all() and (df['use_focal_loss'] == df['focal_extracted']).all()}")

# Get best results by validation accuracy
best_results = df.loc[df.groupby('run_name')['val_acc'].idxmax()]
best_results = best_results.sort_values('val_acc', ascending=False)

print(f"\n=== TOP 10 BEST CONFIGURATIONS ===")
print(best_results[['run_name', 'epoch', 'lr', 'dropout', 'weight_decay', 'use_focal_loss', 'train_acc', 'val_acc', 'overfitting_gap']].head(10).to_string(index=False))

# Analyze overfitting
print(f"\n=== OVERFITTING ANALYSIS ===")
overfitting_stats = df.groupby('run_name').agg({
    'overfitting_gap': ['mean', 'max', 'min'],
    'val_acc': 'max',
    'train_acc': 'max'
}).round(3)

overfitting_stats.columns = ['avg_overfitting_gap', 'max_overfitting_gap', 'min_overfitting_gap', 'best_val_acc', 'best_train_acc']
overfitting_stats = overfitting_stats.sort_values('best_val_acc', ascending=False)

print("Top 10 configurations by validation accuracy:")
print(overfitting_stats.head(10))

# Find configurations with minimal overfitting
min_overfitting = overfitting_stats.sort_values('avg_overfitting_gap')
print(f"\nTop 10 configurations with minimal overfitting:")
print(min_overfitting.head(10))

# Analyze hyperparameter impact
print(f"\n=== HYPERPARAMETER IMPACT ANALYSIS ===")

# Learning rate analysis
lr_analysis = df.groupby('lr').agg({
    'val_acc': ['mean', 'max'],
    'overfitting_gap': 'mean'
}).round(3)
lr_analysis.columns = ['avg_val_acc', 'max_val_acc', 'avg_overfitting_gap']
print("Learning Rate Impact:")
print(lr_analysis.sort_values('max_val_acc', ascending=False))

# Dropout analysis
dropout_analysis = df.groupby('dropout').agg({
    'val_acc': ['mean', 'max'],
    'overfitting_gap': 'mean'
}).round(3)
dropout_analysis.columns = ['avg_val_acc', 'max_val_acc', 'avg_overfitting_gap']
print(f"\nDropout Impact:")
print(dropout_analysis.sort_values('max_val_acc', ascending=False))

# Weight decay analysis
wd_analysis = df.groupby('weight_decay').agg({
    'val_acc': ['mean', 'max'],
    'overfitting_gap': 'mean'
}).round(3)
wd_analysis.columns = ['avg_val_acc', 'max_val_acc', 'avg_overfitting_gap']
print(f"\nWeight Decay Impact:")
print(wd_analysis.sort_values('max_val_acc', ascending=False))

# Focal loss analysis
focal_analysis = df.groupby('use_focal_loss').agg({
    'val_acc': ['mean', 'max'],
    'overfitting_gap': 'mean'
}).round(3)
focal_analysis.columns = ['avg_val_acc', 'max_val_acc', 'avg_overfitting_gap']
print(f"\nFocal Loss Impact:")
print(focal_analysis)

# Find optimal configurations
print(f"\n=== OPTIMAL CONFIGURATION RECOMMENDATIONS ===")

# Best overall performance
best_overall = best_results.iloc[0]
print(f"Best overall validation accuracy: {best_overall['val_acc']:.2f}%")
print(f"Configuration: lr={best_overall['lr']}, dropout={best_overall['dropout']}, weight_decay={best_overall['weight_decay']}, focal_loss={bool(best_overall['use_focal_loss'])}")

# Best with minimal overfitting (overfitting gap < 2%)
min_overfitting_best = overfitting_stats[overfitting_stats['avg_overfitting_gap'] < 2.0].sort_values('best_val_acc', ascending=False)
if len(min_overfitting_best) > 0:
    print(f"\nBest configuration with minimal overfitting (<2% gap):")
    best_min_overfitting = min_overfitting_best.iloc[0]
    print(f"Validation accuracy: {best_min_overfitting['best_val_acc']:.2f}%")
    print(f"Average overfitting gap: {best_min_overfitting['avg_overfitting_gap']:.2f}%")

# Analyze convergence patterns
print(f"\n=== CONVERGENCE ANALYSIS ===")
convergence_analysis = df.groupby('run_name').agg({
    'epoch': 'max',
    'val_acc': ['max', 'last'],
    'train_acc': ['max', 'last']
}).round(3)
convergence_analysis.columns = ['max_epochs', 'best_val_acc', 'final_val_acc', 'best_train_acc', 'final_train_acc']
convergence_analysis['convergence_quality'] = convergence_analysis['best_val_acc'] - convergence_analysis['final_val_acc']

# Find configurations that converged well
well_converged = convergence_analysis[convergence_analysis['convergence_quality'] < 1.0].sort_values('best_val_acc', ascending=False)
print(f"Top 10 well-converged configurations (final acc within 1% of best):")
print(well_converged.head(10)[['best_val_acc', 'final_val_acc', 'convergence_quality', 'max_epochs']])

# Recommendations
print(f"\n=== IMPROVEMENT RECOMMENDATIONS ===")

# 1. Learning rate recommendations
best_lr = lr_analysis.sort_values('max_val_acc', ascending=False).index[0]
print(f"1. Learning Rate: Best performing LR is {best_lr}. Consider fine-tuning around this value.")

# 2. Dropout recommendations
best_dropout = dropout_analysis.sort_values('max_val_acc', ascending=False).index[0]
print(f"2. Dropout: Best performing dropout is {best_dropout}. This suggests the model benefits from {best_dropout*100}% regularization.")

# 3. Weight decay recommendations
best_wd = wd_analysis.sort_values('max_val_acc', ascending=False).index[0]
print(f"3. Weight Decay: Best performing weight decay is {best_wd}. Consider testing values around this.")

# 4. Focal loss recommendations
if focal_analysis.loc[1, 'max_val_acc'] > focal_analysis.loc[0, 'max_val_acc']:
    print("4. Focal Loss: Focal loss shows better performance. Consider using it for imbalanced datasets.")
else:
    print("4. Focal Loss: Standard loss performs better. Focal loss may not be necessary for this dataset.")

# 5. Overfitting recommendations
avg_overfitting = df['overfitting_gap'].mean()
if avg_overfitting > 5:
    print(f"5. Overfitting: High average overfitting gap ({avg_overfitting:.2f}%). Consider:")
    print("   - Increasing dropout rate")
    print("   - Increasing weight decay")
    print("   - Reducing model complexity")
    print("   - Using early stopping")

# 6. Convergence recommendations
avg_epochs = df.groupby('run_name')['epoch'].max().mean()
if avg_epochs < 50:
    print(f"6. Convergence: Average epochs ({avg_epochs:.1f}) suggests models may need more training time.")
    print("   - Consider increasing max epochs")
    print("   - Implement learning rate scheduling")
    print("   - Use patience-based early stopping")

# 7. Hyperparameter combinations
print(f"\n7. Optimal Hyperparameter Combinations:")
print(f"   - Best overall: lr={best_lr}, dropout={best_dropout}, wd={best_wd}")
print(f"   - Best with minimal overfitting: Check configurations with gap < 2%")

# 8. Additional experiments
print(f"\n8. Suggested Additional Experiments:")
print(f"   - Test learning rates: {best_lr/2}, {best_lr*2}")
print(f"   - Test dropout rates: {max(0.1, best_dropout-0.1)}, {min(0.9, best_dropout+0.1)}")
print(f"   - Test weight decay: {best_wd/2}, {best_wd*2}")
print(f"   - Implement learning rate scheduling")
print(f"   - Test data augmentation techniques")
print(f"   - Consider ensemble methods")

# Save detailed analysis
with open('sweep_analysis_report.txt', 'w') as f:
    f.write("=== DETAILED HYPERPARAMETER SWEEP ANALYSIS ===\n\n")
    f.write(f"Total experiments: {len(df)}\n")
    f.write(f"Unique configurations: {df['run_name'].nunique()}\n")
    f.write(f"Epochs per experiment: {df.groupby('run_name')['epoch'].max().mean():.1f}\n\n")
    
    f.write("=== TOP 10 BEST CONFIGURATIONS ===\n")
    f.write(best_results[['run_name', 'epoch', 'lr', 'dropout', 'weight_decay', 'use_focal_loss', 'train_acc', 'val_acc', 'overfitting_gap']].head(10).to_string(index=False))
    f.write("\n\n")
    
    f.write("=== OVERFITTING ANALYSIS ===\n")
    f.write(overfitting_stats.head(10).to_string())
    f.write("\n\n")
    
    f.write("=== HYPERPARAMETER IMPACT ===\n")
    f.write("Learning Rate:\n")
    f.write(lr_analysis.to_string())
    f.write("\n\nDropout:\n")
    f.write(dropout_analysis.to_string())
    f.write("\n\nWeight Decay:\n")
    f.write(wd_analysis.to_string())
    f.write("\n\nFocal Loss:\n")
    f.write(focal_analysis.to_string())

print(f"\nDetailed analysis saved to 'sweep_analysis_report.txt'")
