"""
Main TransPath Model

This module contains the complete TransPath architecture that combines
encoder, transformer, and decoder components for path planning tasks.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .encoder_decoder import Encoder, Decoder
from .transformer import SpatialTransformer
from .pos_embeddings import PosEmbeds


class TransPathModel(nn.Module):
    """
    TransPath: Transformer-based Path Planning Model
    
    Complete architecture for learning heuristic functions or path planning
    through a combination of convolutional encoding, transformer attention,
    and convolutional decoding.
    
    Architecture:
        Input → Encoder → PosEmbeds → SpatialTransformer → PosEmbeds → Decoder → Output
        
    The model processes spatial inputs (maps, start/goal positions) and produces
    spatial outputs (heuristic maps, path predictions, etc.)
    
    Args:
        in_channels: Number of input channels (e.g., map + start + goal)
        out_channels: Number of output channels (e.g., heuristic map)
        hidden_channels: Number of hidden/feature channels
        attn_blocks: Number of transformer blocks in SpatialTransformer
        attn_heads: Number of attention heads in transformer
        cnn_dropout: Dropout probability for CNN components (Encoder/Decoder)
        attn_dropout: Dropout probability for attention components
        downsample_steps: Number of downsampling steps in Encoder (and upsampling in Decoder)
        resolution: Input spatial resolution (height, width)
        use_pos_embeds: Whether to use positional embeddings
        max_v: Maximum coordinate value for positional embeddings (default: 1.0)
        
    Attributes:
        encoder: Encoder module for feature extraction and downsampling
        decoder: Decoder module for feature reconstruction and upsampling
        encoder_pos: Positional embeddings for encoder output
        decoder_pos: Positional embeddings for decoder input
        transformer: Spatial transformer for global attention
        latent_resolution: Resolution of latent space (after encoding)
        
    Examples:
        >>> model = TransPathModel(in_channels=3, out_channels=1)
        >>> x = torch.randn(2, 3, 64, 64)  # (batch, channels, height, width)
        >>> output = model(x)
        >>> output.shape
        torch.Size([2, 1, 64, 64])
        >>> output.min(), output.max()  # Tanh activation in decoder
        (tensor(-1.), tensor(1.))
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        hidden_channels: int = 64,
        attn_blocks: int = 4,
        attn_heads: int = 4,
        cnn_dropout: float = 0.15,
        attn_dropout: float = 0.15,
        downsample_steps: int = 3,
        resolution: Tuple[int, int] = (64, 64),
        use_pos_embeds: bool = True,
        max_v: float = 1.0,
        **kwargs
    ) -> None:
        super().__init__()
        
        # Store configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.attn_blocks = attn_blocks
        self.attn_heads = attn_heads
        self.downsample_steps = downsample_steps
        self.resolution = resolution
        self.use_pos_embeds = use_pos_embeds
        self.max_v = max_v
        
        # Calculate latent space resolution
        self.latent_resolution = (
            resolution[0] // (2 ** downsample_steps),
            resolution[1] // (2 ** downsample_steps)
        )
        
        # Encoder (downsampling)
        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            downsample_steps=downsample_steps,
            dropout=cnn_dropout
        )
        
        # Decoder (upsampling)
        self.decoder = Decoder(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            upsample_steps=downsample_steps,  # Symmetric to encoder
            dropout=cnn_dropout
        )
        
        # Positional embeddings
        if use_pos_embeds:
            self.encoder_pos = PosEmbeds(
                hidden_size=hidden_channels,
                resolution=self.latent_resolution,
                max_v=max_v
            )
            
            self.decoder_pos = PosEmbeds(
                hidden_size=hidden_channels,
                resolution=self.latent_resolution,
                max_v=max_v
            )
        else:
            # Identity modules if no positional embeddings
            self.encoder_pos = nn.Identity()
            self.decoder_pos = nn.Identity()
        
        # Spatial Transformer
        self.transformer = SpatialTransformer(
            in_channels=hidden_channels,
            num_heads=attn_heads,
            depth=attn_blocks,
            dropout=attn_dropout,
            context_dim=None  # Can be extended for cross-attention
        )
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete TransPath model.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
                Typically contains concatenated map, start, and goal channels.
                
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
                Values are in range [-1, 1] due to Tanh activation in decoder.
        """
        # 1. Encode: Extract features and downsample
        x = self.encoder(x)
        
        # 2. Add positional embeddings to encoded features
        x = self.encoder_pos(x)
        
        # 3. Apply spatial transformer for global attention
        x = self.transformer(x)
        
        # 4. Add positional embeddings before decoding
        x = self.decoder_pos(x)
        
        # 5. Decode: Upsample and reconstruct output
        x = self.decoder(x)
        
        return x
    
    def forward_with_intermediates(
        self, 
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass that returns intermediate features for analysis/debugging.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing:
                - 'input': Original input
                - 'encoded': Features after encoder
                - 'encoded_with_pos': Features after encoder + positional embeddings
                - 'transformed': Features after transformer
                - 'decoded_with_pos': Features before decoder + positional embeddings
                - 'output': Final output
        """
        intermediates = {'input': x}
        
        # Encode
        encoded = self.encoder(x)
        intermediates['encoded'] = encoded
        
        # Add positional embeddings
        encoded_with_pos = self.encoder_pos(encoded)
        intermediates['encoded_with_pos'] = encoded_with_pos
        
        # Transform
        transformed = self.transformer(encoded_with_pos)
        intermediates['transformed'] = transformed
        
        # Add positional embeddings before decoder
        decoded_with_pos = self.decoder_pos(transformed)
        intermediates['decoded_with_pos'] = decoded_with_pos
        
        # Decode
        output = self.decoder(decoded_with_pos)
        intermediates['output'] = output
        
        return intermediates
    
    def get_latent_representation(
        self, 
        x: torch.Tensor,
        include_pos_embeds: bool = True
    ) -> torch.Tensor:
        """
        Extract latent representation (features before decoder).
        
        Useful for feature analysis, transfer learning, or multi-task learning.
        
        Args:
            x: Input tensor
            include_pos_embeds: Whether to include positional embeddings
            
        Returns:
            Latent features of shape (batch_size, hidden_channels, H_latent, W_latent)
        """
        x = self.encoder(x)
        
        if include_pos_embeds and self.use_pos_embeds:
            x = self.encoder_pos(x)
            
        x = self.transformer(x)
        
        if include_pos_embeds and self.use_pos_embeds:
            x = self.decoder_pos(x)
            
        return x
    
    def change_resolution(
        self, 
        new_resolution: Tuple[int, int]
    ) -> None:
        """
        Change model resolution dynamically.
        
        Useful for processing inputs of different sizes without retraining.
        
        Args:
            new_resolution: New input resolution (height, width)
            
        Note:
            This only updates positional embeddings. The model architecture
            (convolution sizes, etc.) remains fixed.
        """
        self.resolution = new_resolution
        
        # Update latent resolution
        self.latent_resolution = (
            new_resolution[0] // (2 ** self.downsample_steps),
            new_resolution[1] // (2 ** self.downsample_steps)
        )
        
        # Update positional embeddings if used
        if self.use_pos_embeds:
            self.encoder_pos.change_resolution(self.latent_resolution, self.max_v)
            self.decoder_pos.change_resolution(self.latent_resolution, self.max_v)
    
    def _initialize_weights(self) -> None:
        """
        Initialize model weights using appropriate strategies.
        
        Uses Kaiming initialization for convolutions and linear layers,
        and small values for batch/layer norm weights.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    module.weight, 
                    mode='fan_out', 
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration as dictionary.
        
        Returns:
            Dictionary containing all model hyperparameters
        """
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'hidden_channels': self.hidden_channels,
            'attn_blocks': self.attn_blocks,
            'attn_heads': self.attn_heads,
            'cnn_dropout': 0.15,  # Default value
            'attn_dropout': 0.15,  # Default value
            'downsample_steps': self.downsample_steps,
            'resolution': self.resolution,
            'use_pos_embeds': self.use_pos_embeds,
            'max_v': self.max_v,
            'latent_resolution': self.latent_resolution,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of the model.
        
        Returns:
            String containing model architecture and statistics
        """
        config = self.get_config()
        
        summary_lines = [
            "=" * 30,
            "TransPath Model Summary",
            "=" * 30,
            f"Input:  ({config['in_channels']}, {config['resolution'][0]}, {config['resolution'][1]})",
            f"Output: ({config['out_channels']}, {config['resolution'][0]}, {config['resolution'][1]})",
            f"Latent: ({config['hidden_channels']}, {config['latent_resolution'][0]}, {config['latent_resolution'][1]})",
            "",
            "Components:",
            f"  • Encoder: {self.downsample_steps} downsampling steps",
            f"  • Transformer: {config['attn_blocks']} blocks, {config['attn_heads']} heads",
            f"  • Decoder: {self.downsample_steps} upsampling steps",
            f"  • Positional embeddings: {'Enabled' if config['use_pos_embeds'] else 'Disabled'}",
            "",
            "Parameters:",
            f"  • Total: {config['total_params']:,}",
            f"  • Trainable: {config['trainable_params']:,}",
            "=" * 60,
        ]
        
        return "\n".join(summary_lines)
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"resolution={self.resolution}")


# # Простая factory функция
# def create_transpath_model(
#     model_type: str = "default",
#     **kwargs
# ) -> TransPathModel:
#     """
#     Factory function to create TransPath models with predefined configurations.
    
#     Args:
#         model_type: Type of model configuration
#             - 'default': Standard configuration (64x64 input)
#             - 'small': Smaller model for faster training
#             - 'large': Larger model for better performance
#         **kwargs: Override any default parameters
        
#     Returns:
#         Configured TransPathModel instance
#     """
#     configs = {
#         'default': {
#             'hidden_channels': 64,
#             'attn_blocks': 4,
#             'attn_heads': 4,
#             'downsample_steps': 3,
#         },
#         'small': {
#             'hidden_channels': 32,
#             'attn_blocks': 2,
#             'attn_heads': 2,
#             'downsample_steps': 2,
#         },
#         'large': {
#             'hidden_channels': 128,
#             'attn_blocks': 6,
#             'attn_heads': 8,
#             'downsample_steps': 4,
#         },
#     }
    
#     if model_type not in configs:
#         raise ValueError(f"Unknown model_type: {model_type}")
    
#     config = configs[model_type].copy()
#     config.update(kwargs)
    
#     return TransPathModel(**config)


# # Упрощенная тестовая функция
# def test_transpath_model():
#     """Test the TransPath model."""
#     print("Testing TransPathModel...")
    
#     try:
#         # Test 1: Default model
#         model = TransPathModel()
#         x = torch.randn(2, 3, 64, 64)
#         output = model(x)
        
#         assert output.shape == (2, 1, 64, 64)
#         print("✓ Default model forward pass")
        
#         # Test 2: Without positional embeddings
#         model = TransPathModel(use_pos_embeds=False)
#         x = torch.randn(1, 3, 64, 64)
#         output = model(x)
        
#         assert output.shape == (1, 1, 64, 64)
#         print("✓ Model without positional embeddings")
        
#         # Test 3: Different input channels
#         model = TransPathModel(in_channels=2, out_channels=3)
#         x = torch.randn(1, 2, 64, 64)
#         output = model(x)
        
#         assert output.shape == (1, 3, 64, 64)
#         print("✓ Different input/output channels")
        
#         # Test 4: Factory function
#         model = create_transpath_model('small')
#         assert isinstance(model, TransPathModel)
#         print("✓ Factory function")
        
#         print("\n✅ All TransPathModel tests passed!")
#         return True
        
#     except Exception as e:
#         print(f"\n❌ Error: {e}")
#         import traceback
#         traceback.print_exc()
#         return False


# if __name__ == "__main__":
#     success = test_transpath_model()