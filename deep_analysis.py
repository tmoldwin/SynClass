import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

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

print("=" * 100)
print("DEEP ANALYSIS: WHAT'S ACTUALLY LIMITING YOUR PERFORMANCE")
print("=" * 100)

# 1. PERFORMANCE CEILING ANALYSIS
print("\n1. PERFORMANCE CEILING ANALYSIS")
print("-" * 50)
print(f"Best validation accuracy: {best_results['val_acc'].max():.2f}%")
print(f"Average validation accuracy: {best_results['val_acc'].mean():.2f}%")
print(f"Standard deviation: {best_results['val_acc'].std():.2f}%")

# Check if we're hitting a ceiling
top_10_percent = best_results.head(int(len(best_results) * 0.1))
print(f"Top 10% configurations average: {top_10_percent['val_acc'].mean():.2f}%")
print(f"Performance spread in top 10%: {top_10_percent['val_acc'].max() - top_10_percent['val_acc'].min():.2f}%")

if top_10_percent['val_acc'].max() - top_10_percent['val_acc'].min() < 1.0:
    print("🚨 PERFORMANCE CEILING DETECTED - Hyperparameter tuning won't help much")
    print("   You need architectural changes or more data")
else:
    print("✅ Still room for hyperparameter optimization")

# 2. UNDERFITTING ANALYSIS
print("\n2. UNDERFITTING ANALYSIS")
print("-" * 50)
avg_overfitting = best_results['overfitting_gap'].mean()
print(f"Average overfitting gap: {avg_overfitting:.2f}%")

underfitting_configs = best_results[best_results['overfitting_gap'] < -2.0]
overfitting_configs = best_results[best_results['overfitting_gap'] > 2.0]

print(f"Configurations with underfitting (< -2%): {len(underfitting_configs)}")
print(f"Configurations with overfitting (> 2%): {len(overfitting_configs)}")

if len(underfitting_configs) > len(overfitting_configs):
    print("🚨 UNDERFITTING DOMINATES - Your model is too small or under-regularized")
    print("   RECOMMENDATION: Make model bigger or reduce regularization")
else:
    print("✅ Good balance or overfitting issues")

# 3. LEARNING RATE SENSITIVITY
print("\n3. LEARNING RATE SENSITIVITY")
print("-" * 50)
lr_analysis = best_results.groupby('lr').agg({
    'val_acc': ['mean', 'max', 'std'],
    'overfitting_gap': 'mean'
}).round(3)

print("Learning rate performance:")
for lr in lr_analysis.index:
    max_acc = lr_analysis.loc[lr, ('val_acc', 'max')]
    mean_acc = lr_analysis.loc[lr, ('val_acc', 'mean')]
    std_acc = lr_analysis.loc[lr, ('val_acc', 'std')]
    print(f"  LR {lr}: max={max_acc:.2f}%, mean={mean_acc:.2f}% ± {std_acc:.2f}%")

best_lr = lr_analysis[('val_acc', 'max')].idxmax()
lr_sensitivity = lr_analysis[('val_acc', 'std')].max()
print(f"\nBest LR: {best_lr}")
print(f"LR sensitivity (std): {lr_sensitivity:.2f}%")

if lr_sensitivity > 2.0:
    print("🚨 HIGH LR SENSITIVITY - Model is unstable, needs better optimization")
else:
    print("✅ Stable across learning rates")

# 4. REGULARIZATION ANALYSIS
print("\n4. REGULARIZATION ANALYSIS")
print("-" * 50)

# Dropout analysis
dropout_analysis = best_results.groupby('dropout').agg({
    'val_acc': ['mean', 'max', 'std'],
    'overfitting_gap': 'mean'
}).round(3)

print("Dropout performance:")
for dropout in dropout_analysis.index:
    max_acc = dropout_analysis.loc[dropout, ('val_acc', 'max')]
    mean_acc = dropout_analysis.loc[dropout, ('val_acc', 'mean')]
    print(f"  Dropout {dropout}: max={max_acc:.2f}%, mean={mean_acc:.2f}%")

best_dropout = dropout_analysis[('val_acc', 'max')].idxmax()
print(f"\nBest dropout: {best_dropout}")

if best_dropout > 0.7:
    print("🚨 HIGH DROPOUT NEEDED - Model is overfitting or too complex")
    print("   Consider: simpler architecture, more data, or different regularization")
else:
    print("✅ Reasonable dropout levels")

# 5. CONVERGENCE ANALYSIS
print("\n5. CONVERGENCE ANALYSIS")
print("-" * 50)
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

# Check if models are still improving
still_improving = convergence_data[convergence_data['convergence_quality'] > 1.0]
print(f"Configurations still improving at end: {len(still_improving)}/{len(convergence_data)}")

if len(still_improving) > len(convergence_data) * 0.5:
    print("🚨 MODELS NOT CONVERGED - Need more training or better optimization")
else:
    print("✅ Most models converged")

# 6. ARCHITECTURAL BOTTLENECKS
print("\n6. ARCHITECTURAL BOTTLENECKS")
print("-" * 50)

# Analyze performance vs complexity
performance_vs_complexity = best_results.copy()
performance_vs_complexity['complexity_score'] = (
    performance_vs_complexity['dropout'] * 0.3 + 
    performance_vs_complexity['weight_decay'] * 1000 * 0.7
)

# Check if higher complexity helps
high_complexity = performance_vs_complexity[performance_vs_complexity['complexity_score'] > performance_vs_complexity['complexity_score'].median()]
low_complexity = performance_vs_complexity[performance_vs_complexity['complexity_score'] <= performance_vs_complexity['complexity_score'].median()]

print(f"High complexity avg: {high_complexity['val_acc'].mean():.2f}%")
print(f"Low complexity avg: {low_complexity['val_acc'].mean():.2f}%")

if high_complexity['val_acc'].mean() > low_complexity['val_acc'].mean() + 1.0:
    print("🚨 COMPLEXITY HELPS - Your model is too simple")
    print("   RECOMMENDATION: Make model bigger (more layers, more features)")
else:
    print("✅ Current complexity is appropriate")

# 7. SPECIFIC RECOMMENDATIONS
print("\n7. SPECIFIC RECOMMENDATIONS")
print("-" * 50)

print("BASED ON ANALYSIS:")

# Model size recommendation
if len(underfitting_configs) > len(overfitting_configs) * 2:
    print("🔥 MAKE MODEL BIGGER:")
    print("   - Add more layers (2-3 more)")
    print("   - Increase hidden dimensions by 50-100%")
    print("   - Add residual connections")
    print("   - Consider attention mechanisms")

# Regularization recommendation
if best_dropout > 0.7:
    print("🔥 REDUCE REGULARIZATION:")
    print("   - Lower dropout to 0.3-0.5")
    print("   - Reduce weight decay by 50%")
    print("   - Use label smoothing instead")

# Optimization recommendation
if lr_sensitivity > 2.0 or avg_convergence_quality > 2.0:
    print("🔥 IMPROVE OPTIMIZATION:")
    print("   - Use AdamW optimizer")
    print("   - Implement learning rate warmup")
    print("   - Use gradient clipping")
    print("   - Try different schedulers (cosine, step)")

# Data recommendation
if top_10_percent['val_acc'].max() - top_10_percent['val_acc'].min() < 1.0:
    print("🔥 NEED MORE DATA OR BETTER FEATURES:")
    print("   - Data augmentation (rotation, scaling, noise)")
    print("   - Feature engineering")
    print("   - Collect more training data")
    print("   - Use pre-trained models")

# Architecture recommendation
if high_complexity['val_acc'].mean() > low_complexity['val_acc'].mean() + 1.0:
    print("🔥 ARCHITECTURAL CHANGES:")
    print("   - Switch to ResNet or DenseNet backbone")
    print("   - Add batch normalization")
    print("   - Use skip connections")
    print("   - Implement ensemble methods")

# 8. PRIORITY ORDER
print("\n8. PRIORITY ORDER (Most Impact First)")
print("-" * 50)

priorities = []

if len(underfitting_configs) > len(overfitting_configs) * 2:
    priorities.append("1. MAKE MODEL BIGGER (highest impact)")
if high_complexity['val_acc'].mean() > low_complexity['val_acc'].mean() + 1.0:
    priorities.append("2. CHANGE ARCHITECTURE")
if lr_sensitivity > 2.0:
    priorities.append("3. IMPROVE OPTIMIZATION")
if best_dropout > 0.7:
    priorities.append("4. REDUCE REGULARIZATION")
if top_10_percent['val_acc'].max() - top_10_percent['val_acc'].min() < 1.0:
    priorities.append("5. ADD DATA AUGMENTATION")

if not priorities:
    priorities = ["1. HYPERPARAMETER FINE-TUNING", "2. ENSEMBLE METHODS", "3. ADVANCED TECHNIQUES"]

for priority in priorities:
    print(f"   {priority}")

# 9. EXPECTED IMPROVEMENTS
print("\n9. EXPECTED IMPROVEMENTS")
print("-" * 50)

current_best = best_results['val_acc'].max()
improvements = []

if len(underfitting_configs) > len(overfitting_configs) * 2:
    improvements.append("Bigger model: +3-5%")
if high_complexity['val_acc'].mean() > low_complexity['val_acc'].mean() + 1.0:
    improvements.append("Better architecture: +2-4%")
if lr_sensitivity > 2.0:
    improvements.append("Better optimization: +1-2%")
if best_dropout > 0.7:
    improvements.append("Reduced regularization: +1-2%")

if not improvements:
    improvements = ["Hyperparameter tuning: +1-2%", "Ensemble: +2-3%", "Data augmentation: +1-3%"]

print("Expected improvements:")
for improvement in improvements:
    print(f"   • {improvement}")

# Calculate expected improvement more conservatively
total_expected = current_best + len(improvements) * 2  # Assume 2% per improvement
print(f"\nConservative target: {total_expected:.1f}%")
print(f"Optimistic target: {total_expected + 2:.1f}%")

print("\n" + "=" * 100)
print("SUMMARY: WHAT TO DO RIGHT NOW")
print("=" * 100)

if len(underfitting_configs) > len(overfitting_configs) * 2:
    print("🚨 YOUR MODEL IS TOO SMALL - Make it bigger first")
elif high_complexity['val_acc'].mean() > low_complexity['val_acc'].mean() + 1.0:
    print("🚨 YOUR ARCHITECTURE IS TOO SIMPLE - Change the backbone")
elif lr_sensitivity > 2.0:
    print("🚨 YOUR OPTIMIZATION IS UNSTABLE - Fix the training")
else:
    print("✅ Your model is reasonably sized, focus on data and techniques")

print("=" * 100)
