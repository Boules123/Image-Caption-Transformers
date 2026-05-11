"""
Complete Image Caption Generator Model
Combines CNN encoder and Transformer decoder
"""

import torch
import torch.nn as nn
from encoder import CNNEncoder
from decoder import TransformerDecoder


class ImageCaptionModel(nn.Module):
    """
    Complete image captioning model
    Combines CNN encoder and Transformer decoder with cross-attention
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.encoder = CNNEncoder(config)
        self.decoder = TransformerDecoder(config)
        
    def forward(self, images, captions):
        features = self.encoder(images)  # (B, encoder_dim)
        
        logits = self.decoder(captions, features)  # (B, T, vocab_size)
        
        return logits
    
    def compute_loss(self, images, captions, pad_idx=0):
        logits = self.forward(images, captions[:, :-1])  # Exclude last token
        
        B, T, V = logits.shape
        logits = logits.reshape(B * T, V)
        targets = captions[:, 1:].reshape(B * T)  # Exclude first token (<SOS>)
        
        loss = nn.functional.cross_entropy(
            logits, 
            targets, 
            ignore_index=pad_idx,
            reduction='mean'
        )
        
        return loss
    
    @torch.no_grad()
    def generate(self, images, max_len=20, start_token=1, end_token=2, 
                 top_k=0, temperature=1.0):

        features = self.encoder(images)
        
        generated = self.decoder.generate(
            features, 
            max_len=max_len,
            start_token=start_token,
            end_token=end_token,
            top_k=top_k,
            temperature=temperature
        )
        
        return generated
    
    
    def count_parameters(self, trainable_only=True):
        """Count trainable parameters"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def load_model(self, path):
        """Load model weights from checkpoint"""
        checkpoint = torch.load(path, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        

