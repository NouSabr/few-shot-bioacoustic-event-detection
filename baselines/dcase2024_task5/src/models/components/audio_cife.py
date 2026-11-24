# CIFE design adapted from https://github.com/SissiW/QSFormer (Wang et al., 2023).
# Provenance and release checks: repository-root THIRD_PARTY_NOTICES.md.
"""
AudioCIFE: Cross-scale Interactive Feature Enhancement for Audio

This module implements a Transformer-based cross-scale feature enhancement mechanism
specifically designed for audio feature maps. It enhances target features by attending
to source features from a different scale using cross-attention.

Key Features:
- Preserves spatial dimensions (H, W) throughout the process
- Maintains channel dimensions (no reduction)
- Uses Transformer encoder-decoder architecture for cross-scale attention
- Includes dimension validation for robust operation

Architecture:
1. Source features are processed through a Transformer encoder (self-attention)
2. Target features are enhanced via Transformer decoder (cross-attention with encoded source)
3. Output maintains exact same shape as target input

Date: 2026-04-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


class AudioCIFE(nn.Module):
    """
    Cross-scale Interactive Feature Enhancement module for audio features.
    
    This module enhances target features by incorporating information from source features
    at a different scale using Transformer-based cross-attention mechanism.
    
    The module is designed to work with audio feature maps from ResNet-style architectures,
    where features have shape (Batch, Channels, Height, Width).
    
    Args:
        channels (int): Number of channels in both source and target features. Default: 128
        d_model (int): Dimension of the Transformer model (embedding dimension). 
                       Should match channels for direct feature processing. Default: 128
        nhead (int): Number of attention heads in Transformer. Must divide d_model evenly. Default: 8
        num_encoder_layers (int): Number of Transformer encoder layers. Default: 2
        num_decoder_layers (int): Number of Transformer decoder layers. Default: 2
        dim_feedforward (int): Dimension of feedforward network in Transformer. Default: 256
        dropout (float): Dropout probability. Default: 0.1
        activation (str): Activation function ('relu' or 'gelu'). Default: 'relu'
        
    Input Shapes:
        source_features: (B, 128, 4, 32) - Features from Layer2
        target_features: (B, 128, 4, 32) - Features to be enhanced (Layer3 input)
        
    Output Shape:
        enhanced_features: (B, 128, 4, 32) - Enhanced target features
        
    Example:
        >>> cife = AudioCIFE(channels=128, d_model=128, nhead=8)
        >>> source = torch.randn(4, 128, 4, 32)  # Layer2 output
        >>> target = torch.randn(4, 128, 4, 32)  # Layer3 input
        >>> enhanced = cife(source, target)
        >>> print(enhanced.shape)  # torch.Size([4, 128, 4, 32])
    """
    
    def __init__(
        self,
        channels: int = 128,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_checkpoint: bool = False
    ):
        super(AudioCIFE, self).__init__()
        
        self.channels = channels
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.use_checkpoint = use_checkpoint
        
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        
        if d_model != channels:
            logger.warning(
                f"d_model ({d_model}) != channels ({channels}). "
                f"This may require additional projection layers."
            )
        
        self.source_projection = nn.Sequential(
            nn.Conv2d(channels, d_model, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        
        self.target_projection = nn.Sequential(
            nn.Conv2d(channels, d_model, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        
        self.positional_encoding = nn.Parameter(
            torch.randn(1, d_model, 1, 1) * 0.02
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        self.output_projection = nn.Sequential(
            nn.Conv2d(d_model, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.5)
        
        logger.info(
            f"AudioCIFE initialized: channels={channels}, d_model={d_model}, "
            f"nhead={nhead}, encoder_layers={num_encoder_layers}, "
            f"decoder_layers={num_decoder_layers}, use_checkpoint={use_checkpoint}"
        )
    
    def _validate_input_shapes(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> None:
        """
        Validate that input tensors have the expected shapes.
        
        Args:
            source_features: Source feature tensor
            target_features: Target feature tensor
            
        Raises:
            ValueError: If shapes are invalid
        """
        if source_features.dim() != 4:
            raise ValueError(
                f"source_features must be 4D (B, C, H, W), got {source_features.dim()}D"
            )
        if target_features.dim() != 4:
            raise ValueError(
                f"target_features must be 4D (B, C, H, W), got {target_features.dim()}D"
            )
        
        if source_features.size(0) != target_features.size(0):
            raise ValueError(
                f"Batch size mismatch: source={source_features.size(0)}, "
                f"target={target_features.size(0)}"
            )
        
        if source_features.size(1) != self.channels:
            raise ValueError(
                f"source_features channels mismatch: expected {self.channels}, "
                f"got {source_features.size(1)}"
            )
        if target_features.size(1) != self.channels:
            raise ValueError(
                f"target_features channels mismatch: expected {self.channels}, "
                f"got {target_features.size(1)}"
            )
        
        if source_features.shape[2:] != target_features.shape[2:]:
            raise ValueError(
                f"Spatial dimensions mismatch: source={source_features.shape[2:]}, "
                f"target={target_features.shape[2:]}"
            )
    
    def _reshape_for_transformer(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Reshape feature map from (B, C, H, W) to (B, H*W, C) for Transformer.
        
        Args:
            features: Feature tensor of shape (B, C, H, W)
            
        Returns:
            reshaped: Tensor of shape (B, H*W, C)
            spatial_shape: Original (H, W) for reconstruction
        """
        B, C, H, W = features.shape
        # Reshape: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        reshaped = features.flatten(2).transpose(1, 2)
        return reshaped, (H, W)
    
    def _reshape_from_transformer(
        self,
        features: torch.Tensor,
        spatial_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Reshape from Transformer output (B, H*W, C) back to (B, C, H, W).
        
        Args:
            features: Tensor of shape (B, H*W, C)
            spatial_shape: Target (H, W)
            
        Returns:
            reshaped: Tensor of shape (B, C, H, W)
        """
        B, HW, C = features.shape
        H, W = spatial_shape
        # Reshape: (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        reshaped = features.transpose(1, 2).reshape(B, C, H, W)
        return reshaped
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of AudioCIFE module.
        
        Args:
            source_features: Source features from Layer2, shape (B, 128, 4, 32)
            target_features: Target features for Layer3, shape (B, 128, 4, 32)
            
        Returns:
            enhanced_features: Enhanced target features, shape (B, 128, 4, 32)
            
        Raises:
            ValueError: If input shapes are invalid
        """
        self._validate_input_shapes(source_features, target_features)
        
        target_original = target_features
        B, C, H, W = target_features.shape
        
        source_proj = self.source_projection(source_features)  # (B, d_model, H, W)
        target_proj = self.target_projection(target_features)  # (B, d_model, H, W)
        
        source_proj = source_proj + self.positional_encoding
        target_proj = target_proj + self.positional_encoding
        
        # Reshape for Transformer: (B, C, H, W) -> (B, H*W, C)
        source_seq, spatial_shape = self._reshape_for_transformer(source_proj)
        target_seq, _ = self._reshape_for_transformer(target_proj)
        
        encoded_source = self.transformer_encoder(source_seq)  # (B, H*W, d_model)
        
        enhanced_seq = self.transformer_decoder(
            tgt=target_seq,           # Query: target features
            memory=encoded_source     # Key/Value: encoded source features
        )  # (B, H*W, d_model)
        
        # Reshape back to spatial format: (B, H*W, C) -> (B, C, H, W)
        enhanced_features = self._reshape_from_transformer(enhanced_seq, spatial_shape)
        
        enhanced_features = self.output_projection(enhanced_features)  # (B, channels, H, W)
        
        if enhanced_features.shape != target_original.shape:
            logger.warning(
                f"Output shape mismatch! Expected {target_original.shape}, "
                f"got {enhanced_features.shape}. Applying interpolation."
            )
            # Fallback: interpolate to match target shape
            enhanced_features = F.interpolate(
                enhanced_features,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            # Adjust channels if needed
            if enhanced_features.size(1) != C:
                enhanced_features = F.adaptive_avg_pool2d(enhanced_features, (H, W))
                enhanced_features = enhanced_features[:, :C, :, :]
        
        output = self.residual_weight * enhanced_features + (1 - self.residual_weight) * target_original
        
        return output
    
    def extra_repr(self) -> str:
        """String representation of module parameters."""
        return (
            f"channels={self.channels}, d_model={self.d_model}, nhead={self.nhead}, "
            f"encoder_layers={self.num_encoder_layers}, decoder_layers={self.num_decoder_layers}, "
            f"use_checkpoint={self.use_checkpoint}"
        )


def test_audio_cife():
    """
    Test function to verify AudioCIFE module functionality.
    
    This function creates a sample AudioCIFE module and tests it with
    random input tensors to ensure correct output shapes.
    """
    print("Testing AudioCIFE module...")
    
    cife = AudioCIFE(
        channels=128,
        d_model=128,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1
    )
    
    batch_size = 4
    source = torch.randn(batch_size, 128, 4, 32)  # Layer2 output
    target = torch.randn(batch_size, 128, 4, 32)  # Layer3 input
    
    print(f"Source shape: {source.shape}")
    print(f"Target shape: {target.shape}")
    
    with torch.no_grad():
        output = cife(source, target)
    
    print(f"Output shape: {output.shape}")
    
    assert output.shape == target.shape, f"Shape mismatch! Expected {target.shape}, got {output.shape}"
    assert output.shape == (batch_size, 128, 4, 32), f"Unexpected output shape: {output.shape}"
    
    print("✓ All tests passed!")
    print(f"\nModule parameters: {sum(p.numel() for p in cife.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in cife.parameters() if p.requires_grad):,}")
    
    return cife


if __name__ == "__main__":
    test_audio_cife()
