"""
Training script for Image Caption Generator
Supports GPU training with mixed precision and checkpointing
"""

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import os
import argparse
import numpy as np

from utils.logger import setup_logging
from config import get_config
from dataloader.dataset import build_vocabulary, get_dataloaders
from model.model import ImageCaptionModel
from utils.utils import (
    AverageMeter, save_checkpoint, 
    load_checkpoint, save_training_history
)
from inference import test



def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_epoch(logger, model, train_loader, optimizer, scaler, config):
    model.train()
    loss_meter = AverageMeter()
    grad_norm = 0.0
        
    for batch_idx, (images, captions) in enumerate(train_loader):
        images = images.to(config.device)
        captions = captions.to(config.device)
        
        with autocast():
            loss = model.compute_loss(images, captions, pad_idx=0)
            loss = loss / config.grad_accum_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % config.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        loss_meter.update(loss.item() * config.grad_accum_steps, images.size(0))
    
        if batch_idx % 50 == 0:
            logger.info(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss_meter.avg:.4f} | Grad Norm: {grad_norm:.2f}")
    
    return loss_meter.avg


@torch.no_grad()
def validate(model, val_loader, config):
    model.eval()
    loss_meter = AverageMeter()
    
    
    for images, captions in val_loader:
        images = images.to(config.device)
        captions = captions.to(config.device)
        
        with autocast():
            loss = model.compute_loss(images, captions, pad_idx=0)
        
        loss_meter.update(loss.item(), images.size(0))
            
    return loss_meter.avg


def train(config, logger, train_loader, val_loader, resume_from=None):
    set_seed(42)
    
    

    model = ImageCaptionModel(config)
    model = model.to(config.device)
    
    logger.info(f"num of parameters: {model.count_parameters(trainable_only=True):,}")    

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999))
    
    if config.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.max_epochs,
            eta_min=config.learning_rate * 0.01
        )
    else:
        scheduler = StepLR(
            optimizer,
            step_size=10,
            gamma=config.lr_decay_factor
        )
    
    scaler = GradScaler(enabled=True)
    
    start_epoch = 1
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    if resume_from and os.path.exists(resume_from):
        logger.info(f"\nResuming from checkpoint: {resume_from}")
        start_epoch, best_val_loss = load_checkpoint(
            resume_from, model, optimizer, scheduler, config.device
        )
        start_epoch += 1
    
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, config.max_epochs + 1):
        train_loss = train_epoch(logger, model, train_loader, optimizer, scaler, config)
        val_loss = validate(model, val_loader, config)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        logger.info(f"\nEpoch {epoch}/{config.max_epochs}:")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Learning Rate: {get_lr(optimizer):.6f}")
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, best_path)
            logger.info(f"New best model saved! (Val Loss: {best_val_loss:.4f})")
        
        if epoch % config.save_every == 0:
            latest_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, latest_path)
        
    
    final_path = os.path.join(config.checkpoint_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, scheduler, config.max_epochs, best_val_loss, final_path)
    
    history_path = os.path.join(config.log_dir, 'training_history.json')
    save_training_history(history, history_path)
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Image Caption Generator')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    
    args, unknown = parser.parse_known_args()
    
    
    config_kwargs = {
        'batch_size': args.batch_size,
        'max_epochs': args.epochs,
        'learning_rate': args.lr,
    }
    
    if args.data_dir:
        config_kwargs['data_dir'] = args.data_dir
    
    cfg = get_config(**config_kwargs)
    
    logger = setup_logging(cfg.log_dir)
    
    logger.info(f"starting experiment: {cfg.experiment_name}")
    
    vocab = build_vocabulary(cfg)
    cfg.update_vocab_size(len(vocab))
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    train_loader, val_loader, test_loader = get_dataloaders(cfg, logger, vocab)

    # Train
    train(cfg, logger, train_loader, val_loader, resume_from=args.resume)
    test(test_loader, cfg)