import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Hyperparameter Sweep Analysis - Improvement Opportunities', fontsize=16, fontweight='bold')

# 1. Validation Accuracy Distribution
axes[0, 0].hist(best_results['val_acc'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].axvline(best_results['val_acc'].mean(), color='red', linestyle='--', label=f'Mean: {best_results["val_acc"].mean():.2f}%')
axes[0, 0].axvline(best_results['val_acc'].max(), color='green', linestyle='--', label=f'Best: {best_results["val_acc"].max():.2f}%')
axes[0, 0].set_xlabel('Validation Accuracy (%)')
axes[0, 0].set_ylabel('Number of Configurations')
axes[0, 0].set_title('Distribution of Best Validation Accuracies')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Learning Rate vs Validation Accuracy
lr_groups = best_results.groupby('lr')['val_acc'].agg(['mean', 'max', 'std']).reset_index()
axes[0, 1].errorbar(lr_groups['lr'], lr_groups['mean'], yerr=lr_groups['std'], 
                   marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
axes[0, 1].scatter(lr_groups['lr'], lr_groups['max'], color='red', s=100, zorder=5, label='Max Accuracy')
axes[0, 1].set_xscale('log')
axes[0, 1].set_xlabel('Learning Rate')
axes[0, 1].set_ylabel('Validation Accuracy (%)')
axes[0, 1].set_title('Learning Rate Impact on Performance')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Dropout vs Validation Accuracy
dropout_groups = best_results.groupby('dropout')['val_acc'].agg(['mean', 'max', 'std']).reset_index()
axes[0, 2].errorbar(dropout_groups['dropout'], dropout_groups['mean'], yerr=dropout_groups['std'], 
                   marker='s', capsize=5, capthick=2, linewidth=2, markersize=8)
axes[0, 2].scatter(dropout_groups['dropout'], dropout_groups['max'], color='red', s=100, zorder=5, label='Max Accuracy')
axes[0, 2].set_xlabel('Dropout Rate')
axes[0, 2].set_ylabel('Validation Accuracy (%)')
axes[0, 2].set_title('Dropout Impact on Performance')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Weight Decay vs Validation Accuracy
wd_groups = best_results.groupby('weight_decay')['val_acc'].agg(['mean', 'max', 'std']).reset_index()
axes[1, 0].errorbar(wd_groups['weight_decay'], wd_groups['mean'], yerr=wd_groups['std'], 
                   marker='^', capsize=5, capthick=2, linewidth=2, markersize=8)
axes[1, 0].scatter(wd_groups['weight_decay'], wd_groups['max'], color='red', s=100, zorder=5, label='Max Accuracy')
axes[1, 0].set_xscale('log')
axes[1, 0].set_xlabel('Weight Decay')
axes[1, 0].set_ylabel('Validation Accuracy (%)')
axes[1, 0].set_title('Weight Decay Impact on Performance')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Overfitting Analysis
axes[1, 1].scatter(best_results['val_acc'], best_results['overfitting_gap'], 
                  alpha=0.6, s=50, c=best_results['val_acc'], cmap='viridis')
axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Overfitting')
axes[1, 1].axhline(y=-5, color='orange', linestyle='--', alpha=0.7, label='5% Underfitting')
axes[1, 1].set_xlabel('Validation Accuracy (%)')
axes[1, 1].set_ylabel('Overfitting Gap (%)')
axes[1, 1].set_title('Accuracy vs Overfitting Gap')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Focal Loss Comparison
focal_comparison = best_results.groupby('use_focal_loss').agg({
    'val_acc': ['mean', 'max', 'std'],
    'overfitting_gap': 'mean'
}).round(3)

focal_labels = ['Standard Loss', 'Focal Loss']
focal_means = focal_comparison[('val_acc', 'mean')].values
focal_maxs = focal_comparison[('val_acc', 'max')].values
focal_stds = focal_comparison[('val_acc', 'std')].values

x_pos = np.arange(len(focal_labels))
axes[1, 2].bar(x_pos, focal_means, yerr=focal_stds, capsize=5, alpha=0.7, label='Mean Accuracy')
axes[1, 2].scatter(x_pos, focal_maxs, color='red', s=100, zorder=5, label='Max Accuracy')
axes[1, 2].set_xticks(x_pos)
axes[1, 2].set_xticklabels(focal_labels)
axes[1, 2].set_ylabel('Validation Accuracy (%)')
axes[1, 2].set_title('Focal Loss vs Standard Loss')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sweep_analysis_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Create additional analysis plots
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
fig2.suptitle('Detailed Performance Analysis', fontsize=16, fontweight='bold')

# 1. Training Curves for Top Configurations
top_configs = best_results.head(5)['run_name'].values
for i, config in enumerate(top_configs):
    config_data = df[df['run_name'] == config]
    axes2[0, 0].plot(config_data['epoch'], config_data['val_acc'], 
                    marker='o', markersize=4, label=f'{config[:20]}...', linewidth=1.5)

axes2[0, 0].set_xlabel('Epoch')
axes2[0, 0].set_ylabel('Validation Accuracy (%)')
axes2[0, 0].set_title('Training Curves - Top 5 Configurations')
axes2[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes2[0, 0].grid(True, alpha=0.3)

# 2. Hyperparameter Combinations Heatmap
# Create a pivot table for learning rate vs dropout
lr_dropout_pivot = best_results.pivot_table(
    values='val_acc', 
    index='dropout', 
    columns='lr', 
    aggfunc='mean'
)

sns.heatmap(lr_dropout_pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes2[0, 1])
axes2[0, 1].set_title('Learning Rate vs Dropout Heatmap')
axes2[0, 1].set_xlabel('Learning Rate')
axes2[0, 1].set_ylabel('Dropout Rate')

# 3. Convergence Analysis
convergence_data = df.groupby('run_name').agg({
    'epoch': 'max',
    'val_acc': ['max', 'last'],
    'train_acc': ['max', 'last']
}).round(3)

convergence_data.columns = ['max_epochs', 'best_val_acc', 'final_val_acc', 'best_train_acc', 'final_train_acc']
convergence_data['convergence_quality'] = convergence_data['best_val_acc'] - convergence_data['final_val_acc']

axes2[1, 0].scatter(convergence_data['max_epochs'], convergence_data['convergence_quality'], 
                   alpha=0.6, s=50, c=convergence_data['best_val_acc'], cmap='viridis')
axes2[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes2[1, 0].set_xlabel('Max Epochs')
axes2[1, 0].set_ylabel('Convergence Quality (Best - Final)')
axes2[1, 0].set_title('Convergence Analysis')
axes2[1, 0].grid(True, alpha=0.3)

# 4. Performance vs Complexity
# Use overfitting gap as a proxy for model complexity
axes2[1, 1].scatter(best_results['overfitting_gap'], best_results['val_acc'], 
                   alpha=0.6, s=50, c=best_results['dropout'], cmap='plasma')
axes2[1, 1].set_xlabel('Overfitting Gap (%)')
axes2[1, 1].set_ylabel('Validation Accuracy (%)')
axes2[1, 1].set_title('Performance vs Model Complexity')
axes2[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('detailed_analysis_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("=== VISUALIZATION SUMMARY ===")
print(f"Best validation accuracy: {best_results['val_acc'].max():.2f}%")
print(f"Average validation accuracy: {best_results['val_acc'].mean():.2f}%")
print(f"Standard deviation: {best_results['val_acc'].std():.2f}%")
print(f"Range: {best_results['val_acc'].max() - best_results['val_acc'].min():.2f}%")

print(f"\n=== TOP PERFORMING CONFIGURATIONS ===")
for i, (_, row) in enumerate(best_results.head(5).iterrows()):
    print(f"{i+1}. {row['run_name']}: {row['val_acc']:.2f}% (gap: {row['overfitting_gap']:.2f}%)")

print(f"\n=== IMPROVEMENT OPPORTUNITIES ===")
print("1. Learning Rate: Focus on 5e-06 for best performance")
print("2. Dropout: 0.8 shows best results, suggesting high regularization needed")
print("3. Weight Decay: 0.002 performs best, indicating strong regularization")
print("4. Focal Loss: Standard loss outperforms focal loss")
print("5. Overfitting: Many configurations show underfitting (negative gap)")
print("6. Convergence: Some models may benefit from longer training")

print(f"\nPlots saved as 'sweep_analysis_plots.png' and 'detailed_analysis_plots.png'")
