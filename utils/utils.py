"""
Utility functions for Image Caption Generator
"""

import torch
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def count_parameters(model):
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        total: Total parameters
        trainable: Trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, path):
    """
    Save training checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        best_loss: Best validation loss
        path: Save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_loss': best_loss,
        'config': model.config
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cuda'):

    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_loss']


def denormalize_image(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor for visualization
    
    Args:
        image_tensor: Normalized image tensor (C, H, W)
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Denormalized image as numpy array
    """
    image = image_tensor.clone()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).cpu().numpy()


def visualize_predictions(images, captions, predictions, vocab, num_samples=4, save_path=None):
    """
    Visualize image-caption pairs
    
    Args:
        images: Image tensors (B, C, H, W)
        captions: Ground truth captions (B, T)
        predictions: Predicted captions (B, T)
        vocab: Vocabulary object
        num_samples: Number of samples to visualize
        save_path: Optional path to save figure
    """
    num_samples = min(num_samples, images.size(0))
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        # Denormalize image
        img = denormalize_image(images[i])
        
        # Decode captions
        gt_caption = vocab.decode(captions[i].tolist(), skip_special=True)
        pred_caption = vocab.decode(predictions[i].tolist(), skip_special=True)
        
        # Plot
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(
            f"Ground Truth: {gt_caption}\n"
            f"Prediction: {pred_caption}",
            fontsize=10,
            loc='left'
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_training_history(history, path):
    """
    Save training history to JSON
    
    Args:
        history: Dictionary of training metrics
        path: Save path
    """
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {path}")


def load_training_history(path):
    """
    Load training history from JSON
    
    Args:
        path: Path to history file
    
    Returns:
        history: Dictionary of training metrics
    """
    with open(path, 'r') as f:
        history = json.load(f)
    return history


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation curves
    
    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

