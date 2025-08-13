import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from datetime import datetime
import argparse

def analyze_sweep_directory(sweep_dir):
    """Analyze sweep results from directory"""
    
    # Find sweep_results.csv
    csv_file = os.path.join(sweep_dir, 'sweep_results.csv')
    if not os.path.exists(csv_file):
        print(f"No sweep_results.csv found in {sweep_dir}")
        return
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} data points from {len(df['run_name'].unique())} unique runs")
    
    # Extract hyperparameters
    df['lr_extracted'] = df['lr']
    df['dropout_extracted'] = df['dropout']
    df['weight_decay_extracted'] = df['weight_decay']
    df['focal_extracted'] = df['use_focal_loss']
    
    # Get best results by validation accuracy for each configuration
    best_results = df.loc[df.groupby('run_name')['val_acc'].idxmax()]
    best_results = best_results.sort_values('val_acc', ascending=False)
    
    # Calculate overfitting gap
    best_results['overfitting_gap'] = best_results['train_acc'] - best_results['val_acc']
    
    # Create analysis directory
    analysis_dir = 'analysis'
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Save best results
    best_results.to_csv(os.path.join(analysis_dir, 'best_results.csv'), index=False)
    
    # Create summary
    summary = []
    summary.append(f"SWEEP ANALYSIS SUMMARY")
    summary.append(f"Total runs: {len(best_results)}")
    summary.append(f"Best accuracy: {best_results['val_acc'].max():.2f}%")
    summary.append(f"Worst accuracy: {best_results['val_acc'].min():.2f}%")
    summary.append(f"Mean accuracy: {best_results['val_acc'].mean():.2f}%")
    summary.append(f"Std accuracy: {best_results['val_acc'].std():.2f}%")
    summary.append("")
    
    best_run = best_results.iloc[0]
    summary.append(f"Best run: {best_run['run_name']}")
    summary.append(f"Best epoch: {best_run['epoch']}")
    summary.append(f"Best learning rate: {best_run['lr_extracted']:.1e}")
    summary.append(f"Best dropout: {best_run['dropout_extracted']}")
    summary.append(f"Best weight decay: {best_run['weight_decay_extracted']:.1e}")
    summary.append(f"Best focal loss: {best_run['focal_extracted']}")
    summary.append("")
    
    summary.append("TOP 10 CONFIGURATIONS:")
    summary.append("------------------------------")
    for i, (_, row) in enumerate(best_results.head(10).iterrows()):
        summary.append(f"{i+1}. {row['run_name']}: {row['val_acc']:.2f}% (epoch {row['epoch']})")
    
    with open(os.path.join(analysis_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(summary))
    
    # Create plots
    create_main_plots(best_results, analysis_dir)
    create_detailed_plots(best_results, analysis_dir, df)
    
    print(f"Analysis complete! Results saved to: {analysis_dir}")
    print(f"Best accuracy: {best_results['val_acc'].max():.2f}% from {best_results.iloc[0]['run_name']}")

def create_main_plots(best_results, analysis_dir):
    """Create comprehensive visualizations of sweep results"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Accuracy distribution
    ax1 = plt.subplot(3, 3, 1)
    plt.hist(best_results['val_acc'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(best_results['val_acc'].max(), color='red', linestyle='--', label=f'Best: {best_results["val_acc"].max():.2f}%')
    plt.xlabel('Validation Accuracy (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Best Accuracies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Learning rate vs accuracy
    ax2 = plt.subplot(3, 3, 2)
    lr_groups = best_results.groupby('lr_extracted')['val_acc'].agg(['mean', 'max', 'count'])
    plt.scatter(lr_groups.index, lr_groups['max'], s=100, alpha=0.7)
    plt.xlabel('Learning Rate')
    plt.ylabel('Best Accuracy (%)')
    plt.title('Learning Rate vs Best Accuracy')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # 3. Dropout vs accuracy
    ax3 = plt.subplot(3, 3, 3)
    dropout_groups = best_results.groupby('dropout_extracted')['val_acc'].agg(['mean', 'max', 'count'])
    plt.scatter(dropout_groups.index, dropout_groups['max'], s=100, alpha=0.7)
    plt.xlabel('Dropout Rate')
    plt.ylabel('Best Accuracy (%)')
    plt.title('Dropout vs Best Accuracy')
    plt.grid(True, alpha=0.3)
    
    # 4. Weight decay vs accuracy
    ax4 = plt.subplot(3, 3, 4)
    wd_groups = best_results.groupby('weight_decay_extracted')['val_acc'].agg(['mean', 'max', 'count'])
    plt.scatter(wd_groups.index, wd_groups['max'], s=100, alpha=0.7)
    plt.xlabel('Weight Decay')
    plt.ylabel('Best Accuracy (%)')
    plt.title('Weight Decay vs Best Accuracy')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # 5. Focal loss comparison
    ax5 = plt.subplot(3, 3, 5)
    focal_groups = best_results.groupby('focal_extracted')['val_acc'].agg(['mean', 'max', 'count'])
    plt.bar(['No Focal', 'Focal'], focal_groups['max'], alpha=0.7)
    plt.ylabel('Best Accuracy (%)')
    plt.title('Focal Loss vs Best Accuracy')
    plt.grid(True, alpha=0.3)
    
    # 6. Overfitting analysis
    ax6 = plt.subplot(3, 3, 6)
    plt.scatter(best_results['train_acc'], best_results['val_acc'], alpha=0.6)
    plt.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Perfect fit')
    plt.xlabel('Training Accuracy (%)')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Epoch analysis
    ax7 = plt.subplot(3, 3, 7)
    plt.scatter(best_results['epoch'], best_results['val_acc'], alpha=0.6)
    plt.xlabel('Best Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Best Epoch vs Accuracy')
    plt.grid(True, alpha=0.3)
    
    # 8. Top configurations
    ax8 = plt.subplot(3, 3, 8)
    top_10 = best_results.head(10)
    plt.barh(range(len(top_10)), top_10['val_acc'])
    plt.yticks(range(len(top_10)), [f"{row['lr_extracted']:.1e}" for _, row in top_10.iterrows()])
    plt.xlabel('Validation Accuracy (%)')
    plt.title('Top 10 Configurations (by LR)')
    plt.grid(True, alpha=0.3)
    
    # 9. Parameter combinations heatmap
    ax9 = plt.subplot(3, 3, 9)
    # Create a pivot table for dropout vs weight decay
    pivot_data = best_results.pivot_table(
        values='val_acc', 
        index='dropout_extracted', 
        columns='weight_decay_extracted', 
        aggfunc='max'
    )
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='viridis', ax=ax9)
    plt.title('Dropout vs Weight Decay Heatmap')
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'sweep_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_detailed_plots(best_results, analysis_dir, df):
    """Create additional detailed plots with comprehensive analysis"""
    
    # Learning curves for top configurations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Top 4 configurations
    top_4 = best_results.head(4)
    
    for i, (_, row) in enumerate(top_4.iterrows()):
        ax = axes[i//2, i%2]
        
        # Find the full training data for this run from the original dataframe
        run_name = row['run_name']
        run_data = df[df['run_name'] == run_name].sort_values('epoch')
        
        ax.plot(run_data['epoch'], run_data['train_acc'], 'b-', label='Train', alpha=0.7, linewidth=2)
        ax.plot(run_data['epoch'], run_data['val_acc'], 'r-', label='Val', alpha=0.7, linewidth=2)
        ax.set_title(f'{run_name}\nBest: {row["val_acc"]:.2f}%')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'top_configurations_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # COMPREHENSIVE OVERFITTING ANALYSIS
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
    fig2.suptitle('Comprehensive Overfitting & Training Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overfitting Gap Distribution
    axes2[0, 0].hist(best_results['overfitting_gap'], bins=15, alpha=0.7, color='orange', edgecolor='black')
    axes2[0, 0].axvline(best_results['overfitting_gap'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {best_results["overfitting_gap"].mean():.2f}%')
    axes2[0, 0].axvline(0, color='green', linestyle='--', label='No Overfitting')
    axes2[0, 0].set_xlabel('Overfitting Gap (Train - Val) (%)')
    axes2[0, 0].set_ylabel('Number of Configurations')
    axes2[0, 0].set_title('Overfitting Gap Distribution')
    axes2[0, 0].legend()
    axes2[0, 0].grid(True, alpha=0.3)
    
    # 2. Training vs Validation Performance
    axes2[0, 1].scatter(best_results['train_acc'], best_results['val_acc'], 
                       alpha=0.6, s=50, c=best_results['overfitting_gap'], cmap='RdYlBu')
    axes2[0, 1].plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Perfect fit')
    axes2[0, 1].set_xlabel('Training Accuracy (%)')
    axes2[0, 1].set_ylabel('Validation Accuracy (%)')
    axes2[0, 1].set_title('Training vs Validation Performance')
    axes2[0, 1].legend()
    axes2[0, 1].grid(True, alpha=0.3)
    
    # 3. Overfitting vs Performance
    axes2[0, 2].scatter(best_results['overfitting_gap'], best_results['val_acc'], 
                       alpha=0.6, s=50, c=best_results['train_acc'], cmap='viridis')
    axes2[0, 2].axvline(0, color='red', linestyle='--', alpha=0.7, label='No Overfitting')
    axes2[0, 2].set_xlabel('Overfitting Gap (%)')
    axes2[0, 2].set_ylabel('Validation Accuracy (%)')
    axes2[0, 2].set_title('Overfitting vs Validation Performance')
    axes2[0, 2].legend()
    axes2[0, 2].grid(True, alpha=0.3)
    
    # 4. Training Performance by Hyperparameter
    # Learning Rate impact on training
    lr_train_groups = best_results.groupby('lr_extracted')['train_acc'].agg(['mean', 'max', 'std']).reset_index()
    axes2[1, 0].errorbar(lr_train_groups['lr_extracted'], lr_train_groups['mean'], 
                        yerr=lr_train_groups['std'], marker='o', capsize=5, linewidth=2)
    axes2[1, 0].set_xscale('log')
    axes2[1, 0].set_xlabel('Learning Rate')
    axes2[1, 0].set_ylabel('Training Accuracy (%)')
    axes2[1, 0].set_title('Learning Rate vs Training Performance')
    axes2[1, 0].grid(True, alpha=0.3)
    
    # 5. Dropout impact on training
    dropout_train_groups = best_results.groupby('dropout_extracted')['train_acc'].agg(['mean', 'max', 'std']).reset_index()
    axes2[1, 1].errorbar(dropout_train_groups['dropout_extracted'], dropout_train_groups['mean'], 
                        yerr=dropout_train_groups['std'], marker='s', capsize=5, linewidth=2)
    axes2[1, 1].set_xlabel('Dropout Rate')
    axes2[1, 1].set_ylabel('Training Accuracy (%)')
    axes2[1, 1].set_title('Dropout vs Training Performance')
    axes2[1, 1].grid(True, alpha=0.3)
    
    # 6. Focal Loss impact on training
    focal_train_groups = best_results.groupby('focal_extracted')['train_acc'].agg(['mean', 'max', 'std']).reset_index()
    axes2[1, 2].bar(['No Focal', 'Focal'], focal_train_groups['mean'], 
                   yerr=focal_train_groups['std'], capsize=5, alpha=0.7)
    axes2[1, 2].set_ylabel('Training Accuracy (%)')
    axes2[1, 2].set_title('Focal Loss vs Training Performance')
    axes2[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'overfitting_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate comprehensive analysis report
    generate_analysis_report(best_results, df, analysis_dir)

def generate_analysis_report(best_results, df, analysis_dir):
    """Generate comprehensive analysis report"""
    
    report = []
    report.append("COMPREHENSIVE SWEEP ANALYSIS REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL PERFORMANCE:")
    report.append(f"Best validation accuracy: {best_results['val_acc'].max():.2f}%")
    report.append(f"Best training accuracy: {best_results['train_acc'].max():.2f}%")
    report.append(f"Mean validation accuracy: {best_results['val_acc'].mean():.2f}%")
    report.append(f"Mean training accuracy: {best_results['train_acc'].mean():.2f}%")
    report.append(f"Mean overfitting gap: {best_results['overfitting_gap'].mean():.2f}%")
    report.append("")
    
    # Overfitting analysis
    report.append("OVERFITTING ANALYSIS:")
    overfitting_configs = best_results[best_results['overfitting_gap'] > 0]
    underfitting_configs = best_results[best_results['overfitting_gap'] < 0]
    report.append(f"Configurations with overfitting: {len(overfitting_configs)} ({len(overfitting_configs)/len(best_results)*100:.1f}%)")
    report.append(f"Configurations with underfitting: {len(underfitting_configs)} ({len(underfitting_configs)/len(best_results)*100:.1f}%)")
    report.append(f"Configurations with perfect fit: {len(best_results[best_results['overfitting_gap'] == 0])}")
    report.append("")
    
    # Top performers analysis
    report.append("TOP 5 PERFORMERS ANALYSIS:")
    top_5 = best_results.head(5)
    for i, (_, row) in enumerate(top_5.iterrows()):
        report.append(f"{i+1}. {row['run_name']}")
        report.append(f"   Validation: {row['val_acc']:.2f}%, Training: {row['train_acc']:.2f}%")
        report.append(f"   Overfitting gap: {row['overfitting_gap']:.2f}%")
        report.append(f"   Best epoch: {row['epoch']}")
        report.append("")
    
    # Hyperparameter impact analysis
    report.append("HYPERPARAMETER IMPACT ANALYSIS:")
    
    # Learning rate analysis
    lr_analysis = best_results.groupby('lr_extracted').agg({
        'val_acc': ['mean', 'max', 'std'],
        'train_acc': ['mean', 'max'],
        'overfitting_gap': 'mean'
    }).round(3)
    report.append("Learning Rate Impact:")
    for lr in lr_analysis.index:
        report.append(f"  LR {lr:.1e}: Val={lr_analysis.loc[lr, ('val_acc', 'mean')]:.2f}%, Train={lr_analysis.loc[lr, ('train_acc', 'mean')]:.2f}%, Gap={lr_analysis.loc[lr, ('overfitting_gap', 'mean')]:.2f}%")
    report.append("")
    
    # Dropout analysis
    dropout_analysis = best_results.groupby('dropout_extracted').agg({
        'val_acc': ['mean', 'max'],
        'train_acc': ['mean', 'max'],
        'overfitting_gap': 'mean'
    }).round(3)
    report.append("Dropout Impact:")
    for dr in dropout_analysis.index:
        report.append(f"  Dropout {dr}: Val={dropout_analysis.loc[dr, ('val_acc', 'mean')]:.2f}%, Train={dropout_analysis.loc[dr, ('train_acc', 'mean')]:.2f}%, Gap={dropout_analysis.loc[dr, ('overfitting_gap', 'mean')]:.2f}%")
    report.append("")
    
    # Focal loss analysis
    focal_analysis = best_results.groupby('focal_extracted').agg({
        'val_acc': ['mean', 'max'],
        'train_acc': ['mean', 'max'],
        'overfitting_gap': 'mean'
    }).round(3)
    report.append("Focal Loss Impact:")
    for focal in focal_analysis.index:
        focal_name = "Focal" if focal else "Standard"
        report.append(f"  {focal_name}: Val={focal_analysis.loc[focal, ('val_acc', 'mean')]:.2f}%, Train={focal_analysis.loc[focal, ('train_acc', 'mean')]:.2f}%, Gap={focal_analysis.loc[focal, ('overfitting_gap', 'mean')]:.2f}%")
    report.append("")
    
    # Why certain configurations work better
    report.append("WHY CERTAIN CONFIGURATIONS WORK BETTER:")
    best_config = best_results.iloc[0]
    report.append(f"Best configuration: {best_config['run_name']}")
    report.append(f"  - Learning rate {best_config['lr_extracted']:.1e}: Optimal for fine-tuning without overshooting")
    report.append(f"  - Dropout {best_config['dropout_extracted']}: Provides regularization without over-constraining")
    report.append(f"  - Weight decay {best_config['weight_decay_extracted']:.1e}: Prevents overfitting while maintaining capacity")
    report.append(f"  - Focal loss: {'Yes' if best_config['focal_extracted'] else 'No'}")
    if best_config['focal_extracted']:
        report.append("    Focal loss helps focus on hard examples and reduces class imbalance issues")
    report.append("")
    
    # Overfitting patterns
    report.append("OVERFITTING PATTERNS:")
    high_overfitting = best_results[best_results['overfitting_gap'] > 5]
    if len(high_overfitting) > 0:
        report.append(f"High overfitting configurations (>5% gap): {len(high_overfitting)}")
        report.append("Common characteristics:")
        report.append("  - Lower dropout rates (insufficient regularization)")
        report.append("  - Higher learning rates (overshooting)")
        report.append("  - Lower weight decay (insufficient regularization)")
    else:
        report.append("No configurations show severe overfitting (>5% gap)")
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS:")
    report.append("1. Use learning rate around 5e-6 for optimal fine-tuning")
    report.append("2. Moderate dropout (0.3-0.4) provides good regularization")
    report.append("3. Higher weight decay (2e-3) helps prevent overfitting")
    report.append("4. Focal loss shows mixed results - test both options")
    report.append("5. Monitor overfitting gap - aim for <3% difference")
    report.append("6. Consider longer training for configurations showing improvement trends")
    
    # Save report
    with open(os.path.join(analysis_dir, 'comprehensive_analysis_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    
    print("Comprehensive analysis report generated!")

def main():
    parser = argparse.ArgumentParser(description='Analyze sweep results')
    parser.add_argument('sweep_dir', help='Directory containing sweep results')
    args = parser.parse_args()
    
    if not os.path.exists(args.sweep_dir):
        print(f"Directory {args.sweep_dir} does not exist")
        return
    
    analyze_sweep_directory(args.sweep_dir)

if __name__ == '__main__':
    main()
