"""
Dataset utilities for Image Caption Generator
Adapted from existing implementation with improvements
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import os
from PIL import Image
from torchvision import transforms as T
import re
from collections import Counter
from typing import List, Tuple


def read_caption_file(captions_file: str) -> List[Tuple[str, str]]:
    with open(captions_file, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")[1:]  # Skip header
    
    pairs = []
    for row in lines:
        if not row.strip():
            continue
        parts = row.split(",", 1)  # Split only on first comma
        if len(parts) == 2:
            img_id, caption = parts
            pairs.append((img_id.strip(), caption.strip()))
    
    return pairs


def tokenize(text: str) -> List[str]:
    words = re.findall(r'\w+', text.lower())
    return words


class Vocabulary:
    """Vocabulary for image captions with special tokens"""
    
    def __init__(self, captions: List[str], min_freq: int = 5):
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        
        freq = Counter(tok for cap in captions for tok in tokenize(cap))
        tokens = sorted(tok for tok, cnt in freq.items() if cnt >= min_freq)
        
        for tok in reversed(self.special_tokens):
            if tok not in tokens:
                tokens.insert(0, tok)
        
        self.itos = {i: tok for i, tok in enumerate(tokens)}
        self.stoi = {tok: i for i, tok in self.itos.items()}
        
        assert self.stoi["<PAD>"] == 0, "PAD token must have index 0"

    def encode(self, text: str) -> List[int]:
        return (
            [self.stoi["<SOS>"]]
            + [self.stoi.get(tok, self.stoi["<UNK>"]) for tok in tokenize(text)]
            + [self.stoi["<EOS>"]]
        )
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if skip_special:
            special_ids = {self.stoi[tok] for tok in self.special_tokens}
            tokens = [self.itos[i] for i in ids if i not in special_ids]
        else:
            tokens = [self.itos[i] for i in ids]
        return " ".join(tokens)
    
    def __len__(self):
        return len(self.itos)
    
    @property
    def pad_idx(self):
        return self.stoi["<PAD>"]
    
    @property
    def sos_idx(self):
        return self.stoi["<SOS>"]
    
    @property
    def eos_idx(self):
        return self.stoi["<EOS>"]
    
    @property
    def unk_idx(self):
        return self.stoi["<UNK>"]


class ImageCaptionDataset(Dataset):
    """Dataset for image-caption pairs"""
    
    def __init__(self, images_dir: str, caption_pairs: List[Tuple[str, str]], 
                 vocab: Vocabulary, transform=None):
        self.images_dir = images_dir
        self.pairs = caption_pairs
        self.vocab = vocab
        self.transform = transform
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_name, caption = self.pairs[idx]
        img_path = os.path.join(self.images_dir, img_name)
        

        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        caption_ids = torch.tensor(self.vocab.encode(caption), dtype=torch.long)
        return img, caption_ids


def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions

def build_vocabulary(config):
    caption_pairs = read_caption_file(config.captions_file)
    captions = [cap for _, cap in caption_pairs]
    vocab = Vocabulary(captions, min_freq=config.min_freq)
    return vocab

def get_dataloaders(config, logger, vocab: Vocabulary):
    caption_pairs = read_caption_file(config.captions_file)
    
    # Split dataset
    total = len(caption_pairs)
    train_n = int(config.train_split * total)
    val_n = int(config.val_split * total)
    test_n = total - train_n - val_n
    
    gen = torch.Generator().manual_seed(42)
    
    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    non_train_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create full dataset
    full_dataset = ImageCaptionDataset(
        config.images_dir, 
        caption_pairs, 
        vocab, 
        transform=None
    )
    
    # Split dataset
    train_ds, val_ds, test_ds = random_split(
        full_dataset, 
        [train_n, val_n, test_n], 
        generator=gen
    )
    
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = non_train_transform
    test_ds.dataset.transform = non_train_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    logger.info(f"train split: {len(train_ds)} samples")
    logger.info(f"val split: {len(val_ds)} samples")
    logger.info(f"test split: {len(test_ds)} samples")
    
    return train_loader, val_loader, test_loader


