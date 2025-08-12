import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from datetime import datetime
import argparse

def analyze_sweep_directory(sweep_dir):
    """Analyze all results from a sweep directory"""
    print(f"Analyzing sweep directory: {sweep_dir}")
    
    # Find all CSV files in the sweep directory
    csv_files = glob.glob(os.path.join(sweep_dir, "**/sweep_results.csv"), recursive=True)
    
    if not csv_files:
        print(f"No sweep_results.csv files found in {sweep_dir}")
        return
    
    # Combine all CSV files
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Add the subdirectory name as a column
            subdir = os.path.basename(os.path.dirname(csv_file))
            df['subdir'] = subdir
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if not all_data:
        print("No valid CSV data found")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Extract hyperparameters from run_name
    def extract_params(run_name):
        parts = run_name.split('_')
        try:
            lr = float(parts[0].replace('lr', '').replace('e-', 'e-'))
            dropout = float(parts[1].replace('dr', ''))
            weight_decay = float(parts[2].replace('wd', '').replace('e-', 'e-'))
            focal = 'focal' in run_name
            return lr, dropout, weight_decay, focal
        except:
            return None, None, None, None
    
    # Add extracted parameters
    params = combined_df['run_name'].apply(extract_params)
    combined_df[['lr_extracted', 'dropout_extracted', 'weight_decay_extracted', 'focal_extracted']] = pd.DataFrame(params.tolist(), index=combined_df.index)
    
    # Get best results by validation accuracy for each configuration
    best_results = combined_df.loc[combined_df.groupby('run_name')['val_acc'].idxmax()]
    best_results = best_results.sort_values('val_acc', ascending=False)
    
    # Create analysis directory
    analysis_dir = os.path.join(sweep_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Save combined data
    combined_df.to_csv(os.path.join(analysis_dir, 'all_sweep_results.csv'), index=False)
    best_results.to_csv(os.path.join(analysis_dir, 'best_results.csv'), index=False)
    
    # Generate summary statistics
    summary_stats = {
        'total_runs': len(best_results),
        'best_accuracy': best_results['val_acc'].max(),
        'worst_accuracy': best_results['val_acc'].min(),
        'mean_accuracy': best_results['val_acc'].mean(),
        'std_accuracy': best_results['val_acc'].std(),
        'best_run': best_results.iloc[0]['run_name'],
        'best_epoch': best_results.iloc[0]['epoch'],
        'best_lr': best_results.iloc[0]['lr_extracted'],
        'best_dropout': best_results.iloc[0]['dropout_extracted'],
        'best_weight_decay': best_results.iloc[0]['weight_decay_extracted'],
        'best_focal': best_results.iloc[0]['focal_extracted']
    }
    
    # Save summary
    with open(os.path.join(analysis_dir, 'summary.txt'), 'w') as f:
        f.write("SWEEP ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total runs: {summary_stats['total_runs']}\n")
        f.write(f"Best accuracy: {summary_stats['best_accuracy']:.2f}%\n")
        f.write(f"Worst accuracy: {summary_stats['worst_accuracy']:.2f}%\n")
        f.write(f"Mean accuracy: {summary_stats['mean_accuracy']:.2f}%\n")
        f.write(f"Std accuracy: {summary_stats['std_accuracy']:.2f}%\n\n")
        f.write(f"Best run: {summary_stats['best_run']}\n")
        f.write(f"Best epoch: {summary_stats['best_epoch']}\n")
        f.write(f"Best learning rate: {summary_stats['best_lr']}\n")
        f.write(f"Best dropout: {summary_stats['best_dropout']}\n")
        f.write(f"Best weight decay: {summary_stats['best_weight_decay']}\n")
        f.write(f"Best focal loss: {summary_stats['best_focal']}\n\n")
        
        f.write("TOP 10 CONFIGURATIONS:\n")
        f.write("-" * 30 + "\n")
        for i, row in best_results.head(10).iterrows():
            f.write(f"{i+1}. {row['run_name']}: {row['val_acc']:.2f}% (epoch {row['epoch']})\n")
    
    # Create visualizations
    create_sweep_visualizations(best_results, analysis_dir)
    
    print(f"Analysis complete! Results saved to: {analysis_dir}")
    print(f"Best accuracy: {summary_stats['best_accuracy']:.2f}% from {summary_stats['best_run']}")

def create_sweep_visualizations(best_results, analysis_dir):
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
    
    # Create additional detailed plots
    create_detailed_plots(best_results, analysis_dir)

def create_detailed_plots(best_results, analysis_dir):
    """Create additional detailed plots"""
    
    # Learning curves for top configurations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Top 4 configurations
    top_4 = best_results.head(4)
    
    for i, (_, row) in enumerate(top_4.iterrows()):
        ax = axes[i//2, i%2]
        
        # Find the full training data for this run
        run_name = row['run_name']
        run_data = best_results[best_results['run_name'] == run_name].sort_values('epoch')
        
        ax.plot(run_data['epoch'], run_data['train_acc'], 'b-', label='Train', alpha=0.7)
        ax.plot(run_data['epoch'], run_data['val_acc'], 'r-', label='Val', alpha=0.7)
        ax.set_title(f'{run_name}\nBest: {row["val_acc"]:.2f}%')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'top_configurations_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

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
