"""
CNN Encoder for Image Caption Generator
Uses ResNet50 pretrained on ImageNet to extract image features
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    """
    CNN-based image encoder using ResNet50
    Extracts 2048-dimensional features from images
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        
        self.feature_proj = nn.Linear(2048, config.encoder_dim)
        
        self._freeze_params()
        
        self._init_params()
    
    def _freeze_params(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
    
    def _init_params(self):
        nn.init.xavier_uniform_(self.feature_proj.weight)
        if self.feature_proj.bias is not None:
            nn.init.constant_(self.feature_proj.bias, 0)
    
    def forward(self, images):
        with torch.no_grad():  
            x = self.resnet(images)
        
        x = x.view(x.size(0), -1)  
        x = self.feature_proj(x)   
        
        return x
    
    def unfreeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = True
