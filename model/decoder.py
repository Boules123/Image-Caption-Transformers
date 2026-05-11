"""
Transformer Decoder for Image Caption Generator
Implements a transformer decoder with cross-attention to image features
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention using scaled_dot_product_attention (PyTorch 2.x)
    Prevents attending to future tokens in the sequence
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dim must be divisible by num heads"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd) # Q, K, V projection
        
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_drop = nn.Dropout(config.dropout)
    
    def forward(self, x):
        B, L, C = x.size()
        
        # q, k, v projections 
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        
        q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, L, D)
        
        y = y.transpose(1, 2).contiguous().view(B, L, C)  # (B, L, E)
        
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.drop = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer decoder block with self-attention and cross-attention
    """
    
    def __init__(self, config):
        super().__init__()
        self.self_attn = CausalSelfAttention(config)
        
        self.cross_attn = nn.MultiheadAttention(
            config.n_embd, 
            config.n_head, 
            dropout=config.dropout, 
            batch_first=True
        )
        
        self.mlp = MLP(config)
        
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
    
    def forward(self, x, img_emb):
        x = x + self.self_attn(self.ln1(x))
        
        attn_out, _ = self.cross_attn(
            self.ln2(x), 
            img_emb, 
            img_emb, 
            need_weights=False
        )
        x = x + attn_out
        
        x = x + self.mlp(self.ln3(x))
        
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for image captioning
    Generates captions conditioned on image features
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # tok embedd 
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        # pos embedd + pos image tok
        self.position_embedding = nn.Embedding(config.block_size + 1, config.n_embd)
        
        self.feature_proj = nn.Linear(config.encoder_dim, config.n_embd)
        
        # learnable image tok (anchor for cross-attention)
        self.image_token = nn.Parameter(torch.randn(1, 1, config.n_embd))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, captions, features):
        B, T = captions.size()
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block size {self.config.block_size}"
        
        img_proj = self.feature_proj(features).unsqueeze(1)  # (B, 1, E)
        
        img_token = self.image_token.expand(B, -1, -1)  # (B, 1, E)
        img_emb = img_token + img_proj  # (B, 1, E)
        
        cap_emb = self.token_embedding(captions)  # (B, T, E)
        
        x = torch.cat([img_emb, cap_emb], dim=1)  # (B, T+1, E)
        
        positions = torch.arange(0, T + 1, device=captions.device)
        positions = positions.unsqueeze(0).expand(B, T + 1)
        pos_emb = self.position_embedding(positions)
        
        x = self.dropout(x + pos_emb)
        
        for block in self.blocks:
            x = block(x, img_emb)
        
        x = self.ln_f(x)
        
        x = x[:, 1:, :]  # (B, T, E)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        return logits
    
    @torch.no_grad()
    def generate(self, features, max_len=20, start_token=1, end_token=2, 
                 top_k=0, temperature=1.0):
        B = features.size(0)
        device = features.device
        
        # Start with <SOS> token
        generated = torch.full((B, 1), start_token, dtype=torch.long, device=device)
        
        for _ in range(max_len):
            logits = self.forward(generated, features)  # (B, T, V)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # (B, V)
            
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(logits, top_k)
                probs = F.softmax(topk_vals, dim=-1)
                next_token = topk_idx.gather(-1, torch.multinomial(probs, num_samples=1))
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if torch.all(next_token.squeeze(-1) == end_token):
                break
        
        return generated
