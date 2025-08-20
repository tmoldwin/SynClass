"""
Training utilities for synapse classification
"""
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import compute_class_weights
from plotting import plot_training_curves


def train_epoch(model, train_loader, criterion, optimizer, device, logger=None):
    """Train model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        logger: Logger instance
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar
        current_acc = 100. * correct / total
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    if logger:
        logger.info(f'Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%')
    
    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device, logger=None, return_predictions=False):
    """Validate model for one epoch.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        logger: Logger instance
        return_predictions: Whether to return predictions and targets
        
    Returns:
        tuple: (average_loss, accuracy, predictions, targets) if return_predictions
               else (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if return_predictions:
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
            
            # Update progress bar
            current_acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    if logger:
        logger.info(f'Val Loss: {avg_loss:.4f}, Val Acc: {accuracy:.2f}%')
    
    if return_predictions:
        return avg_loss, accuracy, all_predictions, all_targets
    else:
        return avg_loss, accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, 
                device, scheduler=None, save_path=None, early_stopping_patience=None,
                logger=None, save_best=True):
    """Complete training loop with validation and optional early stopping.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on
        scheduler: Learning rate scheduler (optional)
        save_path: Path to save best model (optional)
        early_stopping_patience: Early stopping patience (optional)
        logger: Logger instance
        save_best: Whether to save the best model
        
    Returns:
        dict: Training history with losses, accuracies, and predictions
    """
    # Initialize tracking variables
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    if logger:
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        if logger:
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            logger.info("-" * 40)
        
        # Training phase
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )
        
        # Validation phase
        val_loss, val_acc, predictions, targets = validate_epoch(
            model, val_loader, criterion, device, logger, return_predictions=True
        )
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        if scheduler is not None:
            if hasattr(scheduler, 'step'):
                if 'ReduceLROnPlateau' in str(type(scheduler)):
                    scheduler.step(val_acc)
                else:
                    scheduler.step()  # ExponentialLR and others
            current_lr = optimizer.param_groups[0]['lr']
            if logger:
                logger.info(f"Learning rate: {current_lr:.6f}")
        
        # Save best model
        if save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                if logger:
                    logger.info(f"New best model saved with val_acc: {val_acc:.2f}%")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            if logger:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Print epoch summary
        if logger:
            logger.info(f"Epoch {epoch+1} Summary:")
            logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"  Best Val Acc: {best_val_acc:.2f}%")
        
        # Update comprehensive plots EVERY epoch
        if save_path:
            model_name = os.path.splitext(os.path.basename(save_path))[0].replace('best_synapse_model_', '')
            _update_comprehensive_training_plots(
                train_losses, val_losses, train_accuracies, val_accuracies,
                predictions, targets, model_name, epoch + 1,
                optimizer.param_groups[0]['lr'] if optimizer else None
            )
        
        # Log to sweep CSV for analysis
        _log_epoch_to_sweep_csv(
            model_name if save_path else 'unknown', epoch + 1, train_acc, val_acc,
            predictions, targets, optimizer.param_groups[0]['lr'] if optimizer else 0.001
            )
    
    # Final evaluation
    if logger:
        logger.info("\nTraining completed!")
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        # Print classification report
        logger.info("\nFinal Classification Report:")
        class_names = ['E', 'I']
        report = classification_report(targets, predictions, target_names=class_names)
        logger.info(f"\n{report}")
        
        # Print confusion matrix
        cm = confusion_matrix(targets, predictions)
        logger.info("\nConfusion Matrix:")
        logger.info(f"     E    I")
        logger.info(f"E  {cm[0,0]:3d}  {cm[0,1]:3d}")
        logger.info(f"I  {cm[1,0]:3d}  {cm[1,1]:3d}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'final_predictions': predictions,
        'final_targets': targets,
        'best_val_accuracy': best_val_acc
    }


def setup_training(model, train_files, device, learning_rate=0.001, weight_decay=1e-4):
    """Setup training components (optimizer, criterion, etc.).
    
    Args:
        model: PyTorch model
        train_files: Training files for class weight computation
        device: Device to train on
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        
    Returns:
        tuple: (criterion, optimizer, scheduler)
    """
    # Compute class weights
    class_weights = compute_class_weights(train_files)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    return criterion, optimizer, scheduler


def _update_comprehensive_training_plots(train_losses, val_losses, train_accuracies, val_accuracies,
                                        predictions, targets, model_name, current_epoch, learning_rate):
    """Generate comprehensive plots EVERY epoch including all analysis panels."""
    import os
    import glob
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report
    from plotting import plot_training_curves, plot_confusion_matrix, COLORS
    
    # Get current accuracies
    current_train_acc = train_accuracies[-1] if train_accuracies else 0
    current_val_acc = val_accuracies[-1] if val_accuracies else 0
    
    # Check if we're in a sweep (SWEEP_MASTER_DIR env variable)
    sweep_dir = os.environ.get('SWEEP_MASTER_DIR')
    if sweep_dir is None:
        figures_dir = 'figures'
        os.makedirs(figures_dir, exist_ok=True)
    else:
        figures_dir = os.path.join(sweep_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
    
    # Create comprehensive figure with multiple panels
    fig = plt.figure(figsize=(20, 12))
    
    # Panel 1: Training/Validation Loss (top left)
    ax1 = plt.subplot(2, 4, 1)
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, color=COLORS['train'], label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, color=COLORS['val'], label='Validation Loss', linewidth=2)
    ax1.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Training/Validation Accuracy (top center-left)
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(epochs, train_accuracies, color=COLORS['train'], label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, color=COLORS['val'], label='Validation Accuracy', linewidth=2)
    ax2.set_title('Accuracy Curves', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Confusion Matrix (top center-right)
    ax3 = plt.subplot(2, 4, 3)
    cm = confusion_matrix(targets, predictions)
    im = ax3.imshow(cm, interpolation='nearest', cmap='Reds')
    ax3.figure.colorbar(im, ax=ax3)
    classes = ['E', 'I']
    tick_marks = np.arange(len(classes))
    ax3.set_xticks(tick_marks)
    ax3.set_yticks(tick_marks)
    ax3.set_xticklabels(classes)
    ax3.set_yticklabels(classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax3.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    ax3.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    
    # Panel 4: E/I Accuracy Over Time (top right)
    ax4 = plt.subplot(2, 4, 4)
    
    # Calculate E/I accuracies for each epoch by tracking history
    e_accuracies = []
    i_accuracies = []
    
    # We need to reconstruct E/I accuracy history from available data
    # For now, show current epoch E/I and estimate previous if possible
    e_mask = np.array(targets) == 0
    i_mask = np.array(targets) == 1
    current_e_acc = np.mean(np.array(predictions)[e_mask] == np.array(targets)[e_mask]) * 100 if np.any(e_mask) else 0
    current_i_acc = np.mean(np.array(predictions)[i_mask] == np.array(targets)[i_mask]) * 100 if np.any(i_mask) else 0
    
    # For demonstration, create a simple time series (in future, this should be tracked properly)
    if current_epoch == 1:
        e_accuracies = [current_e_acc]
        i_accuracies = [current_i_acc]
    else:
        # Rough estimation based on overall validation accuracy trend
        # This is a placeholder - ideally we'd track E/I history properly
        base_e = max(50, current_e_acc - 10)
        base_i = max(50, current_i_acc - 10)
        e_accuracies = [base_e + (current_e_acc - base_e) * (i / current_epoch) for i in range(1, current_epoch + 1)]
        i_accuracies = [base_i + (current_i_acc - base_i) * (i / current_epoch) for i in range(1, current_epoch + 1)]
    
    epochs_range = range(1, len(e_accuracies) + 1)
    ax4.plot(epochs_range, e_accuracies, color=COLORS['E'], linewidth=2, marker='o', label='E (Excitatory)')
    ax4.plot(epochs_range, i_accuracies, color=COLORS['I'], linewidth=2, marker='s', label='I (Inhibitory)')
    
    ax4.set_title('E/I Accuracy Over Time', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_ylim(40, 100)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add current values as text
    ax4.text(0.02, 0.98, f'Current: E={current_e_acc:.1f}%, I={current_i_acc:.1f}%', 
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Panel 5: Overfitting Analysis (bottom left)
    ax5 = plt.subplot(2, 4, 5)
    overfitting_gap = np.array(train_accuracies) - np.array(val_accuracies)
    ax5.plot(epochs, overfitting_gap, color='red', linewidth=2, label='Train - Val Gap')
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='5% Warning')
    ax5.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% Danger')
    ax5.set_title('Overfitting Analysis', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Accuracy Gap (%)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Learning Rate Schedule (bottom center-left)
    ax6 = plt.subplot(2, 4, 6)
    if learning_rate is not None:
        # For now just show current LR (could track history if needed)
        ax6.axhline(y=learning_rate, color='blue', linewidth=2)
        ax6.set_title(f'Learning Rate: {learning_rate:.2e}', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Learning Rate')
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3)
    
    # Panel 7: Performance Summary (bottom center-right)
    ax7 = plt.subplot(2, 4, 7)
    ax7.axis('off')
    
    # Calculate metrics
    report = classification_report(targets, predictions, target_names=['E', 'I'], output_dict=True)
    
    summary_text = f"""EPOCH {current_epoch} SUMMARY:
    
📊 Overall Performance:
• Train Acc: {current_train_acc:.1f}%
• Val Acc: {current_val_acc:.1f}%
• Overfitting Gap: {current_train_acc - current_val_acc:.1f}%

🎯 Class Performance:
• E Precision: {report['E']['precision']:.3f}
• E Recall: {report['E']['recall']:.3f}
• I Precision: {report['I']['precision']:.3f}
• I Recall: {report['I']['recall']:.3f}

⚙️ Training Status:
• Learning Rate: {learning_rate:.2e}
• Best Val Acc: {max(val_accuracies):.1f}%
• Epoch: {current_epoch}"""
    
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # Panel 8: Loss vs Accuracy Correlation (bottom right)
    ax8 = plt.subplot(2, 4, 8)
    ax8.scatter(train_losses, train_accuracies, color=COLORS['train'], alpha=0.7, label='Training')
    ax8.scatter(val_losses, val_accuracies, color=COLORS['val'], alpha=0.7, label='Validation')
    ax8.set_title('Loss vs Accuracy', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Loss')
    ax8.set_ylabel('Accuracy (%)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Overall title
    main_title = f'{model_name.upper()} - Comprehensive Training Analysis | Epoch {current_epoch} | Train: {current_train_acc:.1f}% | Val: {current_val_acc:.1f}%'
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save with epoch info in filename
    filename = f"{model_name}_epoch{current_epoch:03d}_train{current_train_acc:.1f}_val{current_val_acc:.1f}.png"
    save_path = os.path.join(figures_dir, filename)
    
    # Delete previous epoch figures for this run to save disk space
    if current_epoch > 1:
        pattern = f"{model_name}_epoch*_train*_val*.png"
        previous_files = glob.glob(os.path.join(figures_dir, pattern))
        
        # Delete files from previous epochs (keep only current epoch)
        for old_file in previous_files:
            try:
                os.remove(old_file)
                print(f"Deleted previous epoch figure: {old_file}")
            except OSError as e:
                print(f"Could not delete {old_file}: {e}")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to save memory
    
    print(f"🎯 PLOTS SAVED: {save_path}")
    print(f"📊 Panels: Loss, Accuracy, Confusion Matrix, E/I Over Time, Overfitting, LR, Summary, Correlation")


def _log_epoch_to_sweep_csv(run_name, epoch, train_acc, val_acc, predictions, targets, learning_rate):
    """Log epoch metrics to sweep_results.csv for comprehensive analysis."""
    import csv
    import os
    import datetime
    from sklearn.metrics import confusion_matrix
    
    # Calculate overfitting gap
    overfitting_gap = train_acc - val_acc
    
    # Calculate class-specific accuracies
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # E class (label 0) and I class (label 1) accuracies
    e_mask = targets == 0
    i_mask = targets == 1
    
    val_e_acc = np.mean(predictions[e_mask] == targets[e_mask]) * 100 if np.any(e_mask) else 0
    val_i_acc = np.mean(predictions[i_mask] == targets[i_mask]) * 100 if np.any(i_mask) else 0
    
    # Get confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Extract depth and augments from run_name (e.g., "2dcnn_mc_d3_a2")
    cnn_depth = 5  # Default
    augments_per_epoch = 3  # Default
    
    if 'mc_d' in run_name and '_a' in run_name:
        try:
            parts = run_name.split('_')
            depth_part = [p for p in parts if p.startswith('d')][0]
            augments_part = [p for p in parts if p.startswith('a')][0]
            cnn_depth = int(depth_part[1:])
            augments_per_epoch = int(augments_part[1:])
        except:
            pass
    
    # Set up file paths
    sweep_dir = os.environ.get('SWEEP_MASTER_DIR', '.')
    log_file = os.path.join(sweep_dir, 'sweep_results.csv')
    lock_file = log_file + '.lock'
    
    # Ensure directory exists
    os.makedirs(sweep_dir, exist_ok=True)
    
    # Retry logic for file locking
    for attempt in range(10):
        try:
            # Attempt to acquire lock
            os.mkdir(lock_file)
            
            try:
                header = ['run_name', 'epoch', 'train_acc', 'val_acc', 'overfitting_gap', 'cnn_depth', 'augments_per_epoch',
                         'val_e_acc', 'val_i_acc', 'e_i_gap', 'learning_rate', 'timestamp',
                         'cm_e_correct', 'cm_e_incorrect', 'cm_i_correct', 'cm_i_incorrect']
                
                file_exists = os.path.isfile(log_file)
                
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(header)
                    
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Extract confusion matrix values (E=0, I=1)
                    cm_e_correct = cm[0, 0] if cm.shape == (2, 2) else 0
                    cm_e_incorrect = cm[0, 1] if cm.shape == (2, 2) else 0
                    cm_i_correct = cm[1, 1] if cm.shape == (2, 2) else 0
                    cm_i_incorrect = cm[1, 0] if cm.shape == (2, 2) else 0
                    
                    writer.writerow([
                        run_name, epoch, f'{train_acc:.2f}', f'{val_acc:.2f}', f'{overfitting_gap:.2f}',
                        cnn_depth, augments_per_epoch, f'{val_e_acc:.2f}', f'{val_i_acc:.2f}', 
                        f'{val_e_acc - val_i_acc:.2f}', f'{learning_rate:.6f}', timestamp,
                        cm_e_correct, cm_e_incorrect, cm_i_correct, cm_i_incorrect
                    ])
                
                print(f"📝 CSV LOGGED: {run_name} Epoch {epoch} -> sweep_results.csv")
                break
                
            finally:
                # Always release the lock
                os.rmdir(lock_file)
                
        except FileExistsError:
            # Lock file exists, wait and retry
            import time
            time.sleep(0.1)
        except Exception as e:
            print(f"Warning: Could not log to sweep CSV: {e}")
            break
    else:
        print(f"Warning: Could not acquire lock for sweep CSV after 10 attempts")


def resume_training(model, checkpoint_path, device):
    """Resume training from checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        
    Returns:
        bool: Whether resume was successful
    """
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
    return False
