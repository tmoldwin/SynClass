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
                    scheduler.step()
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
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    return criterion, optimizer, scheduler


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
