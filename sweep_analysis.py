import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data(csv_file):
    """Load and prepare the sweep results data"""
    print("Loading sweep results data...")
    df = pd.read_csv(csv_file)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create a unique identifier for each run
    df['run_id'] = df['run_name']
    
    print(f"Loaded {len(df)} data points from {df['run_name'].nunique()} unique runs")
    print(f"Depth range: {df['cnn_depth'].min()}-{df['cnn_depth'].max()}")
    print(f"Width range: {df['cnn_width'].min()}-{df['cnn_width'].max()}")
    
    return df

def get_best_runs(df, n=5):
    """Get the best n runs based on final validation accuracy"""
    # Get final epoch for each run
    final_epochs = df.groupby('run_name')['epoch'].max()
    
    # Get final validation accuracy for each run
    best_runs = []
    for run_name in final_epochs.index:
        final_epoch = final_epochs[run_name]
        final_val_acc = df[(df['run_name'] == run_name) & (df['epoch'] == final_epoch)]['val_acc'].iloc[0]
        best_runs.append({
            'run_name': run_name,
            'final_val_acc': final_val_acc,
            'depth': df[df['run_name'] == run_name]['cnn_depth'].iloc[0],
            'width': df[df['run_name'] == run_name]['cnn_width'].iloc[0]
        })
    
    # Sort by final validation accuracy and get top n
    best_runs_df = pd.DataFrame(best_runs)
    best_runs_df = best_runs_df.sort_values('final_val_acc', ascending=False).head(n)
    
    return best_runs_df

def create_comprehensive_analysis(df, save_path='comprehensive_sweep_analysis.png'):
    """Create a comprehensive figure with all 5 analysis panels"""
    print("\nCreating comprehensive analysis figure with all panels...")
    
    # Get best runs
    best_runs_df = get_best_runs(df, n=5)
    
    # Get final epoch for each run
    final_data = df.groupby('run_name').apply(
        lambda x: x.loc[x['epoch'].idxmax()]
    ).reset_index(drop=True)
    
    # Create the main figure with subplots
    fig = plt.figure(figsize=(24, 18))
    
    # Panel 1: Training curves (top left)
    ax1 = plt.subplot(3, 4, 1)
    colors = plt.cm.Set3(np.linspace(0, 1, len(best_runs_df)))
    
    for i, (_, run) in enumerate(best_runs_df.iterrows()):
        run_data = df[df['run_name'] == run['run_name']]
        ax1.plot(run_data['epoch'], run_data['train_acc'], 
                color=colors[i], linewidth=2, alpha=0.8,
                label=f"d{run['depth']}_w{run['width']}")
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Accuracy (%)')
    ax1.set_title('Best 5 Training Curves', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(50, 75)
    
    # Panel 2: Validation curves (top center)
    ax2 = plt.subplot(3, 4, 2, sharey=ax1)
    
    for i, (_, run) in enumerate(best_runs_df.iterrows()):
        run_data = df[df['run_name'] == run['run_name']]
        ax2.plot(run_data['epoch'], run_data['val_acc'], 
                color=colors[i], linewidth=2, alpha=0.8,
                label=f"d{run['depth']}_w{run['width']}")
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Best 5 Validation Curves', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(50, 75)
    
    # Panel 3: Correlation scatter (top right)
    ax3 = plt.subplot(3, 4, 3)
    
    # Create scatter plot with size based on depth and color based on width
    scatter = ax3.scatter(final_data['train_acc'], final_data['val_acc'], 
                        c=final_data['cnn_width'], s=final_data['cnn_depth']*20, 
                        alpha=0.7, cmap='viridis')
    
    # Add trend line
    z = np.polyfit(final_data['train_acc'], final_data['val_acc'], 1)
    p = np.poly1d(z)
    ax3.plot(final_data['train_acc'], p(final_data['train_acc']), "r--", alpha=0.8, linewidth=2)
    
    # Calculate correlation and R-squared
    correlation = final_data['train_acc'].corr(final_data['val_acc'])
    r_squared = correlation ** 2
    
    # Calculate regression equation
    slope = z[0]
    intercept = z[1]
    
    ax3.set_xlabel('Training Accuracy (%)')
    ax3.set_ylabel('Validation Accuracy (%)')
    ax3.set_title(f'Train vs Val Correlation\nr={correlation:.3f}, R²={r_squared:.3f}', fontsize=12, fontweight='bold')
    
    # Add regression equation annotation
    ax3.text(0.05, 0.95, f'y = {slope:.3f}x + {intercept:.3f}', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3, shrink=0.8)
    cbar.set_label('CNN Width')
    
    # Panel 4: Sample counts (top right)
    ax4 = plt.subplot(3, 4, 4)
    ax4.axis('off')
    
    # Get real sample counts from the sweep results CSV if available
    import os
    import pandas as pd
    
    # Try to get sample counts from sweep results CSV
    sweep_csv_path = 'sweep_results.csv'
    if os.path.exists(sweep_csv_path):
        try:
            sweep_df = pd.read_csv(sweep_csv_path)
            # Check if the new columns exist
            if 'train_e_total' in sweep_df.columns and 'train_i_total' in sweep_df.columns:
                # Get the first row to extract sample counts
                first_row = sweep_df.iloc[0]
                train_e_count = int(first_row['train_e_total'])
                train_i_count = int(first_row['train_i_total'])
                val_e_count = int(first_row['val_e_total'])
                val_i_count = int(first_row['val_i_total'])
                
                sample_info = f"""
    DATASET INFO:
    
    Training Set:
    • Excitatory (E): {train_e_count:,} samples
    • Inhibitory (I): {train_i_count:,} samples
    • Total: {train_e_count + train_i_count:,} samples
    
    Validation Set:
    • Excitatory (E): {val_e_count:,} samples  
    • Inhibitory (I): {val_i_count:,} samples
    • Total: {val_e_count + val_i_count:,} samples
    
    Class Balance:
    • Training: {train_e_count/(train_e_count+train_i_count)*100:.1f}% E, {train_i_count/(train_e_count+train_i_count)*100:.1f}% I
    • Validation: {val_e_count/(val_e_count+val_i_count)*100:.1f}% E, {val_i_count/(val_e_count+val_i_count)*100:.1f}% I
    """
            else:
                sample_info = """
    DATASET INFO:
    
    Training Set:
    • Excitatory (E): [Data not found]
    • Inhibitory (I): [Data not found]
    • Total: [Data not found]
    
    Validation Set:
    • Excitatory (E): [Data not found]  
    • Inhibitory (I): [Data not found]
    • Total: [Data not found]
    
    Class Balance:
    • Training: [Data not found]
    • Validation: [Data not found]
    """
        except Exception as e:
            sample_info = f"""
    DATASET INFO:
    
    Training Set:
    • Excitatory (E): [Error reading data]
    • Inhibitory (I): [Error reading data]
    • Total: [Error reading data]
    
    Validation Set:
    • Excitatory (E): [Error reading data]  
    • Inhibitory (I): [Error reading data]
    • Total: [Error reading data]
    
    Class Balance:
    • Training: [Error reading data]
    • Validation: [Error reading data]
    
    Error: {str(e)}
    """
    else:
        sample_info = """
    DATASET INFO:
    
    Training Set:
    • Excitatory (E): [Data not found]
    • Inhibitory (I): [Data not found]
    • Total: [Data not found]
    
    Validation Set:
    • Excitatory (E): [Data not found]  
    • Inhibitory (I): [Data not found]
    • Total: [Data not found]
    
    Class Balance:
    • Training: [Data not found]
    • Validation: [Data not found]
    """
    
    ax4.text(0.1, 0.9, sample_info, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # Panel 5: Width vs Validation (middle left)
    ax5 = plt.subplot(3, 4, 5)
    
    for depth in sorted(final_data['cnn_depth'].unique()):
        depth_data = final_data[final_data['cnn_depth'] == depth]
        ax5.scatter(depth_data['cnn_width'], depth_data['val_acc'], 
                   label=f'Depth {depth}', s=100, alpha=0.7)
    
    # Add overall regression line
    z = np.polyfit(final_data['cnn_width'], final_data['val_acc'], 1)
    p = np.poly1d(z)
    ax5.plot(final_data['cnn_width'], p(final_data['cnn_width']), "r--", alpha=0.8, linewidth=2)
    
    # Calculate correlation and R-squared
    correlation = final_data['cnn_width'].corr(final_data['val_acc'])
    r_squared = correlation ** 2
    
    ax5.set_xlabel('CNN Width')
    ax5.set_ylabel('Final Validation Accuracy (%)')
    ax5.set_title(f'Width vs Validation Accuracy\nr={correlation:.3f}, R²={r_squared:.3f}', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Depth vs Validation (middle center)
    ax6 = plt.subplot(3, 4, 6)
    
    for width in sorted(final_data['cnn_width'].unique()):
        width_data = final_data[final_data['cnn_width'] == width]
        ax6.scatter(width_data['cnn_depth'], width_data['val_acc'], 
                   label=f'Width {width}', s=100, alpha=0.7)
    
    # Add overall regression line
    z = np.polyfit(final_data['cnn_depth'], final_data['val_acc'], 1)
    p = np.poly1d(z)
    ax6.plot(final_data['cnn_depth'], p(final_data['cnn_depth']), "r--", alpha=0.8, linewidth=2)
    
    # Calculate correlation and R-squared
    correlation = final_data['cnn_depth'].corr(final_data['val_acc'])
    r_squared = correlation ** 2
    
    ax6.set_xlabel('CNN Depth')
    ax6.set_ylabel('Final Validation Accuracy (%)')
    ax6.set_title(f'Depth vs Validation Accuracy\nr={correlation:.3f}, R²={r_squared:.3f}', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # Panel 7: Summary heatmap (middle right)
    ax7 = plt.subplot(3, 4, 7)
    
    # Create pivot table for heatmap
    heatmap_data = final_data.pivot_table(
        values='val_acc', 
        index='cnn_depth', 
        columns='cnn_width', 
        aggfunc='mean'
    )
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax7, cbar_kws={'shrink': 0.8})
    ax7.set_title('Average Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax7.set_xlabel('CNN Width')
    ax7.set_ylabel('CNN Depth')
    
    # Panel 8: Overfitting analysis (middle right)
    ax8 = plt.subplot(3, 4, 8)
    
    overfitting_heatmap = final_data.pivot_table(
        values='overfitting_gap', 
        index='cnn_depth', 
        columns='cnn_width', 
        aggfunc='mean'
    )
    
    sns.heatmap(overfitting_heatmap, annot=True, fmt='.1f', cmap='RdBu_r', ax=ax8, cbar_kws={'shrink': 0.8})
    ax8.set_title('Overfitting Gap (Train-Val)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('CNN Width')
    ax8.set_ylabel('CNN Depth')
    
    # Panel 9: Box plot by width (bottom left)
    ax9 = plt.subplot(3, 4, 9)
    
    width_groups = []
    width_labels = []
    for width in sorted(final_data['cnn_width'].unique()):
        width_data = final_data[final_data['cnn_width'] == width]
        width_groups.append(width_data['val_acc'].values)
        width_labels.append(f'W{width}')
    
    ax9.boxplot(width_groups, labels=width_labels)
    ax9.set_xlabel('CNN Width')
    ax9.set_ylabel('Validation Accuracy (%)')
    ax9.set_title('Validation Accuracy by Width', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    ax9.tick_params(axis='x', rotation=45)
    
    # Panel 10: Box plot by depth (bottom center)
    ax10 = plt.subplot(3, 4, 10)
    
    depth_groups = []
    depth_labels = []
    for depth in sorted(final_data['cnn_depth'].unique()):
        depth_data = final_data[final_data['cnn_depth'] == depth]
        depth_groups.append(depth_data['val_acc'].values)
        depth_labels.append(f'D{depth}')
    
    ax10.boxplot(depth_groups, labels=depth_labels)
    ax10.set_xlabel('CNN Depth')
    ax10.set_ylabel('Validation Accuracy (%)')
    ax10.set_title('Validation Accuracy by Depth', fontsize=12, fontweight='bold')
    ax10.grid(True, alpha=0.3)
    ax10.tick_params(axis='x', rotation=45)
    
    # Panel 11: Performance summary (bottom right)
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    
    # Add performance summary
    best_overall = final_data.loc[final_data['val_acc'].idxmax()]
    min_overfitting = final_data.loc[final_data['overfitting_gap'].idxmin()]
    
    performance_summary = f"""
    PERFORMANCE SUMMARY:
    
    🏆 Best Overall:
    d{best_overall['cnn_depth']}_w{best_overall['cnn_width']}
    {best_overall['val_acc']:.2f}% validation
    
    ⚖️ Best Generalization:
    d{min_overfitting['cnn_depth']}_w{min_overfitting['cnn_width']}
    Gap: {min_overfitting['overfitting_gap']:.2f}%
    
    📊 Statistics:
    • Mean Val Acc: {final_data['val_acc'].mean():.1f}%
    • Std Val Acc: {final_data['val_acc'].std():.1f}%
    • Correlation: {correlation:.3f}
    """
    
    ax11.text(0.1, 0.9, performance_summary, transform=ax11.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # Panel 12: Epochs completed heatmap (bottom right)
    ax12 = plt.subplot(3, 4, 12)
    
    # Calculate epochs completed for each model configuration
    epochs_completed = df.groupby(['cnn_depth', 'cnn_width'])['epoch'].max().reset_index()
    epochs_pivot = epochs_completed.pivot(index='cnn_depth', columns='cnn_width', values='epoch')
    
    # Determine if models finished (assuming 150 epochs is the target)
    target_epochs = 150  # This should match the EPOCHS setting in the training script
    finished_models = epochs_completed[epochs_completed['epoch'] >= target_epochs]
    
    # Create annotations with asterisks for finished models
    annot_data = epochs_pivot.copy()
    for depth in epochs_pivot.index:
        for width in epochs_pivot.columns:
            if not pd.isna(epochs_pivot.loc[depth, width]):
                epochs = int(epochs_pivot.loc[depth, width])
                # Check if this model finished
                is_finished = len(finished_models[(finished_models['cnn_depth'] == depth) & 
                                                (finished_models['cnn_width'] == width)]) > 0
                if is_finished:
                    annot_data.loc[depth, width] = f"{epochs}*"
                else:
                    annot_data.loc[depth, width] = f"{epochs}"
    
    # Create heatmap
    sns.heatmap(epochs_pivot, annot=annot_data, fmt='', cmap='YlOrRd', ax=ax12, 
                cbar_kws={'shrink': 0.8, 'label': 'Epochs Completed'})
    ax12.set_title('Epochs Completed by Model\n(* = Finished Training)', fontsize=12, fontweight='bold')
    ax12.set_xlabel('CNN Width')
    ax12.set_ylabel('CNN Depth')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_runs_df, correlation, heatmap_data

def print_summary_statistics(df, best_runs_df, correlation):
    """Print comprehensive summary statistics"""
    print("\n" + "="*80)
    print("COMPREHENSIVE SWEEP ANALYSIS SUMMARY")
    print("="*80)
    
    # Get final epoch for each run
    final_data = df.groupby('run_name').apply(
        lambda x: x.loc[x['epoch'].idxmax()]
    ).reset_index(drop=True)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   • Total runs: {df['run_name'].nunique()}")
    print(f"   • Total epochs: {len(df)}")
    print(f"   • Depth range: {df['cnn_depth'].min()}-{df['cnn_depth'].max()}")
    print(f"   • Width range: {df['cnn_width'].min()}-{df['cnn_width'].max()}")
    
    print(f"\n🏆 TOP 5 PERFORMING CONFIGURATIONS:")
    for i, (_, run) in enumerate(best_runs_df.iterrows(), 1):
        print(f"   {i}. d{run['depth']}_w{run['width']}: {run['final_val_acc']:.2f}%")
    
    print(f"\n📈 PERFORMANCE STATISTICS:")
    print(f"   • Best validation accuracy: {final_data['val_acc'].max():.2f}%")
    print(f"   • Average validation accuracy: {final_data['val_acc'].mean():.2f}%")
    print(f"   • Std validation accuracy: {final_data['val_acc'].std():.2f}%")
    print(f"   • Train-Val correlation: {correlation:.3f}")
    
    # Calculate regression statistics for all relationships
    train_val_corr = final_data['train_acc'].corr(final_data['val_acc'])
    train_val_r2 = train_val_corr ** 2
    width_val_corr = final_data['cnn_width'].corr(final_data['val_acc'])
    width_val_r2 = width_val_corr ** 2
    depth_val_corr = final_data['cnn_depth'].corr(final_data['val_acc'])
    depth_val_r2 = depth_val_corr ** 2
    
    print(f"\n📊 REGRESSION STATISTICS:")
    print(f"   • Train vs Val: r={train_val_corr:.3f}, R²={train_val_r2:.3f}")
    print(f"   • Width vs Val: r={width_val_corr:.3f}, R²={width_val_r2:.3f}")
    print(f"   • Depth vs Val: r={depth_val_corr:.3f}, R²={depth_val_r2:.3f}")
    
    print(f"\n🔍 DEPTH ANALYSIS:")
    depth_stats = final_data.groupby('cnn_depth')['val_acc'].agg(['mean', 'std', 'max']).round(2)
    for depth, stats in depth_stats.iterrows():
        print(f"   • Depth {depth}: Mean={stats['mean']}%, Std={stats['std']}%, Max={stats['max']}%")
    
    print(f"\n⏱️  TRAINING PROGRESS:")
    # Calculate epochs completed for each model
    epochs_completed = df.groupby(['cnn_depth', 'cnn_width'])['epoch'].max().reset_index()
    target_epochs = 150  # Should match EPOCHS in training script
    
    finished_models = epochs_completed[epochs_completed['epoch'] >= target_epochs]
    total_models = len(epochs_completed)
    completed_models = len(finished_models)
    
    print(f"   • Total model configurations: {total_models}")
    print(f"   • Completed training ({target_epochs} epochs): {completed_models} ({completed_models/total_models*100:.1f}%)")
    print(f"   • Incomplete training: {total_models - completed_models}")
    
    if len(finished_models) > 0:
        print(f"   • Completed models:")
        for _, model in finished_models.iterrows():
            print(f"     - d{model['cnn_depth']}_w{model['cnn_width']}: {model['epoch']} epochs")
    
    if len(epochs_completed[epochs_completed['epoch'] < target_epochs]) > 0:
        print(f"   • Incomplete models:")
        incomplete = epochs_completed[epochs_completed['epoch'] < target_epochs]
        for _, model in incomplete.iterrows():
            print(f"     - d{model['cnn_depth']}_w{model['cnn_width']}: {model['epoch']} epochs")
    
    print(f"\n🔍 WIDTH ANALYSIS:")
    width_stats = final_data.groupby('cnn_width')['val_acc'].agg(['mean', 'std', 'max']).round(2)
    for width, stats in width_stats.iterrows():
        print(f"   • Width {width}: Mean={stats['mean']}%, Std={stats['std']}%, Max={stats['max']}%")
    
    print(f"\n⚖️ OVERFITTING ANALYSIS:")
    print(f"   • Average overfitting gap: {final_data['overfitting_gap'].mean():.2f}%")
    print(f"   • Min overfitting gap: {final_data['overfitting_gap'].min():.2f}%")
    print(f"   • Max overfitting gap: {final_data['overfitting_gap'].max():.2f}%")
    
    # Find configurations with minimal overfitting
    min_overfitting = final_data.loc[final_data['overfitting_gap'].idxmin()]
    print(f"   • Best generalization: d{min_overfitting['cnn_depth']}_w{min_overfitting['cnn_width']} "
          f"(gap: {min_overfitting['overfitting_gap']:.2f}%)")
    
    print(f"\n💡 RECOMMENDATIONS:")
    
    # Best overall performance
    best_overall = final_data.loc[final_data['val_acc'].idxmax()]
    print(f"   • Best overall: d{best_overall['cnn_depth']}_w{best_overall['cnn_width']} "
          f"({best_overall['val_acc']:.2f}%)")
    
    # Best depth for different widths
    print(f"   • Best depth by width:")
    for width in sorted(final_data['cnn_width'].unique()):
        width_data = final_data[final_data['cnn_width'] == width]
        best_depth = width_data.loc[width_data['val_acc'].idxmax()]
        print(f"     - Width {width}: Depth {best_depth['cnn_depth']} ({best_depth['val_acc']:.2f}%)")
    
    # Best width for different depths
    print(f"   • Best width by depth:")
    for depth in sorted(final_data['cnn_depth'].unique()):
        depth_data = final_data[final_data['cnn_depth'] == depth]
        best_width = depth_data.loc[depth_data['val_acc'].idxmax()]
        print(f"     - Depth {depth}: Width {best_width['cnn_width']} ({best_width['val_acc']:.2f}%)")
    
    # Efficiency analysis (performance vs model size)
    print(f"\n⚡ EFFICIENCY ANALYSIS:")
    final_data['model_size'] = final_data['cnn_depth'] * final_data['cnn_width']
    efficiency_data = final_data.groupby('model_size')['val_acc'].agg(['mean', 'max']).round(2)
    print(f"   • Performance vs model size (depth × width):")
    for size, stats in efficiency_data.iterrows():
        print(f"     - Size {size}: Mean={stats['mean']}%, Max={stats['max']}%")
    
    print("\n" + "="*80)

def main():
    """Main analysis function"""
    print("Starting comprehensive sweep analysis...")
    
    # Load data
    df = load_and_prepare_data('sweep_results.csv')
    
    # Create comprehensive analysis figure
    best_runs_df, correlation, heatmap_data = create_comprehensive_analysis(df)
    
    # Print summary
    print_summary_statistics(df, best_runs_df, correlation)
    
    print("\nAnalysis complete! Comprehensive figure saved as 'comprehensive_sweep_analysis.png'")

if __name__ == "__main__":
    main()
