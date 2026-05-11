"""
Configuration file for Image Caption Generator using Transformers
"""

import torch
from dataclasses import dataclass
import os

@dataclass
class Config:    
    experiment_name: str = "image_caption_transformer_experiment"
    
    # Model Architecture
    vocab_size: int = 10000  
    n_embd: int = 512        
    n_layer: int = 6         
    n_head: int = 8          
    encoder_dim: int = 2048  
    dropout: float = 0.1     
    block_size: int = 64     
    
    # Training Parameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_epochs: int = 30
    warmup_epochs: int = 2
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum_steps: int = 2  
    
    # Learning Rate Scheduler
    lr_scheduler: str = "cosine"  # "cosine" or "step"
    lr_decay_factor: float = 0.1
    lr_decay_epochs: list = None  
    
    # Dataset Parameters
    data_dir: str = "..."
    images_dir: str = "..."  # Will be set in __post_init__
    captions_file: str = "..."  # Will be set in __post_init__
    min_freq: int = 5  
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Data Augmentation
    image_size: int = 224
    random_crop: bool = True
    random_flip: bool = True
    
    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1  
    keep_last_n: int = 3
    
    # Logging
    log_dir: str = "logs"
    log_every: int = 100  
    
    # Generation
    max_gen_len: int = 20
    temperature: float = 1.0
    top_k: int = 0  
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Special Tokens
    pad_token: str = "<PAD>"
    sos_token: str = "<SOS>"
    eos_token: str = "<EOS>"
    unk_token: str = "<UNK>"
    
    # post-init processing
    def __post_init__(self):
        if self.images_dir is None:
            self.images_dir = os.path.join(self.data_dir, "data", "flickr8k", "Images")
        if self.captions_file is None:
            self.captions_file = os.path.join(self.data_dir, "captions.txt")
        
        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Validate splits
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
            "Train, val, and test splits must sum to 1.0"
        
        # Validate architecture
        assert self.n_embd % self.n_head == 0, \
            f"Embedding dimension {self.n_embd} must be divisible by number of heads {self.n_head}"
        
    
    def update_vocab_size(self, vocab_size: int):
        self.vocab_size = vocab_size
        print(f"  Vocabulary size updated to: {vocab_size}")


def get_config(**kwargs):
    config = Config(**kwargs)
    return config