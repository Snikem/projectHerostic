import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange
from typing import Optional, Tuple, Union

class ResNetBlock(nn.Module):
    """
    Residual block with GroupNorm, SiLU activation, and dropout.
    
    This block implements a standard ResNet-style residual connection with
    two convolutional layers, normalization, and activation functions.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (defaults to in_channels)
        dropout: Dropout probability applied before the second convolution
    
    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels
        norm1: First GroupNorm layer
        norm2: Second GroupNorm layer
        conv1: First convolutional layer
        conv2: Second convolutional layer
        dropout: Dropout layer
        silu: SiLU activation function
        idConv: 1x1 convolution for residual connection (if channel dimensions change had being changed)
    
    Examples:
        >>> block = ResNetBlock(in_channels=64, out_channels=128, dropout=0.1)
        >>> x = torch.randn(4, 64, 32, 32)
        >>> output = block(x)
        >>> output.shape
        torch.Size([4, 128, 32, 32])
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: Optional[int] = None, 
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        # Normalization layers
        self.norm1 = nn.GroupNorm(
            num_groups=32, 
            num_channels=in_channels, 
            eps=1e-6, 
            affine=True
        )
        self.norm2 = nn.GroupNorm(
            num_groups=32, 
            num_channels=out_channels, 
            eps=1e-6, 
            affine=True
        )

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Activation and regularization
        self.dropout = nn.Dropout(dropout)
        self.silu = nn.SiLU()
        
        # Identity convolution for residual connection (if channels change)
        if in_channels != out_channels:
            self.idConv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
            )
        else:
            self.idConv = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet block.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        h = x  # Keep original input for residual connection

        # First convolution block
        h = self.norm1(h)
        h = self.silu(h)
        h = self.conv1(h)

        # Second convolution block
        h = self.norm2(h)
        h = self.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Residual connection
        x_res = self.idConv(x)
        
        return x_res + h


class Downsample(nn.Module):
    """
    Downsampling block using nearest-neighbor interpolation followed by convolution.
    
    This block reduces spatial dimensions by a factor of 2 while maintaining
    the number of channels. It uses interpolation rather than strided convolution
    for smoother downsampling.
    
    Args:
        in_channels: Number of input/output channels
        scale_factor: Factor by which to downsample (default: 0.5 for 2x reduction)
        mode: Interpolation mode ('nearest', 'bilinear', 'bicubic')
        kernel_size: Size of convolutional kernel
        padding: Padding for convolution
    
    Attributes:
        conv: Convolutional layer applied after downsampling
        scale_factor: Downsampling factor
        mode: Interpolation mode
    
    Examples:
        >>> downsample = Downsample(in_channels=64)
        >>> x = torch.randn(4, 64, 32, 32)
        >>> output = downsample(x)
        >>> output.shape
        torch.Size([4, 64, 16, 16])
    """
    
    def __init__(
        self, 
        in_channels: int, 
        scale_factor: float = 0.5,
        mode: str = "nearest",
        kernel_size: int = 3,
        padding: int = 1
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the downsampling block.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Downsampled tensor of shape (batch_size, channels, height//2, width//2)
        """
        # Downsample using interpolation
        x = F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode=self.mode
        )
        # Apply convolution
        x = self.conv(x)
        return x
    
    def extra_repr(self) -> str:
        """Extra representation string for debugging."""
        return f"in_channels={self.conv.in_channels}, scale_factor={self.scale_factor}, mode={self.mode}"


class Upsample(nn.Module):
    """
    Upsampling block using nearest-neighbor interpolation followed by convolution.
    
    This block increases spatial dimensions by a factor of 2 while maintaining
    the number of channels.
    
    Args:
        in_channels: Number of input/output channels
        scale_factor: Factor by which to upsample (default: 2.0 for 2x increase)
        mode: Interpolation mode ('nearest', 'bilinear', 'bicubic')
        kernel_size: Size of convolutional kernel
        padding: Padding for convolution
    
    Attributes:
        conv: Convolutional layer applied after upsampling
        scale_factor: Upsampling factor
        mode: Interpolation mode
    
    Examples:
        >>> upsample = Upsample(in_channels=64)
        >>> x = torch.randn(4, 64, 16, 16)
        >>> output = upsample(x)
        >>> output.shape
        torch.Size([4, 64, 32, 32])
    """
    
    def __init__(
        self, 
        in_channels: int, 
        scale_factor: float = 2.0,
        mode: str = "nearest",
        kernel_size: int = 3,
        padding: int = 1
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the upsampling block.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Upsampled tensor of shape (batch_size, channels, height*2, width*2)
        """
        # Upsample using interpolation
        x = F.interpolate(
            x, 
            scale_factor=self.scale_factor, 
            mode=self.mode
        )
        # Apply convolution
        x = self.conv(x)
        return x
    
    def extra_repr(self) -> str:
        """Extra representation string for debugging."""
        return f"in_channels={self.conv.in_channels}, scale_factor={self.scale_factor}, mode={self.mode}"


class FeedForward(nn.Module):
    """
    Feed-forward network (MLP) block commonly used in Transformer architectures.
    
    Implements a two-layer MLP with GELU activation and dropout:
    Linear -> GELU -> Dropout -> Linear
    
    Args:
        in_channels: Dimension of input features
        hidden_channels: Dimension of hidden layer
        out_channels: Dimension of output features (defaults to in_channels)
        dropout: Dropout probability applied after activation
    
    Attributes:
        linear1: First linear layer (in_channels -> hidden_channels)
        linear2: Second linear layer (hidden_channels -> out_channels)
        gelu: GELU activation function
        dropout: Dropout layer
    
    Examples:
        >>> ff = FeedForward(in_channels=512, hidden_channels=2048, dropout=0.1)
        >>> x = torch.randn(4, 512)
        >>> output = ff(x)
        >>> output.shape
        torch.Size([4, 512])
    """
    
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        out_channels: Optional[int] = None, 
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        # Linear layers
        self.linear1 = nn.Linear(
            in_features=in_channels,
            out_features=hidden_channels
        )
        self.linear2 = nn.Linear(
            in_features=hidden_channels,
            out_features=self.out_channels
        )

        # Activation and regularization
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, ..., in_channels)
            
        Returns:
            Output tensor of shape (batch_size, ..., out_channels)
        """
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
    def extra_repr(self) -> str:
        """Extra representation string for debugging."""
        return (f"in_channels={self.in_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"out_channels={self.out_channels}")


class BasicTransformerBlock(nn.Module):
    """
    Basic Transformer block with self-attention and cross-attention.
    
    This block implements a standard Transformer layer with:
    1. Self-attention with pre-LayerNorm
    2. Cross-attention with pre-LayerNorm (optional)
    3. Feed-forward network with pre-LayerNorm
    
    Architecture: x -> LN -> Self-Attn -> Add -> LN -> Cross-Attn -> Add -> LN -> FFN -> Add
    
    Args:
        embed_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability for attention and FFN
        context_dim: Dimension of context for cross-attention (if None, uses self-attention)
        
    Attributes:
        attn1: Self-attention module
        attn2: Cross-attention module (or self-attention if context_dim is None)
        ff: Feed-forward network
        norm1: LayerNorm before self-attention
        norm2: LayerNorm before cross-attention
        norm3: LayerNorm before FFN
        
    Examples:
        >>> block = BasicTransformerBlock(embed_dim=512, num_heads=8)
        >>> x = torch.randn(2, 256, 512)  # (batch, seq_len, embed_dim)
        >>> output = block(x)
        >>> output.shape
        torch.Size([2, 256, 512])
        
        >>> # With context
        >>> context = torch.randn(2, 128, 256)
        >>> output = block(x, context=context)
        >>> output.shape
        torch.Size([2, 256, 512])
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.3,
        context_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        
        # Self-attention
        self.attn1 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention (or self-attention if no context provided)
        self.attn2 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True
        )
        
        # Feed-forward network
        self.ff = FeedForward(
            in_channels=embed_dim,
            hidden_channels=embed_dim * 4,
            dropout=dropout
        )
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Store dimensions for reference
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_dim = context_dim

    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embed_dim)
            context: Context tensor for cross-attention of shape 
                    (batch_size, context_length, context_dim). If None, 
                    cross-attention defaults to self-attention.
                    
        Returns:
            Output tensor of shape (batch_size, sequence_length, embed_dim)
        """
        # Self-attention block
        h1 = self.norm1(x)
        h1, _ = self.attn1(
            query=h1,
            key=h1,
            value=h1,
            need_weights=False
        )
        x = h1 + x  # Residual connection
        
        # Cross-attention block
        h2 = self.norm2(x)
        
        # If no context provided, use self-attention (context = h2)
        if context is None:
            context = h2
            
        h2, _ = self.attn2(
            query=h2,
            key=context,
            value=context,
            need_weights=False
        )
        x = h2 + x  # Residual connection
        
        # Feed-forward block
        h3 = self.norm3(x)
        h3 = self.ff(h3)
        x = h3 + x  # Residual connection
        
        return x
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f"embed_dim={self.embed_dim}, "
                f"num_heads={self.num_heads}, "
                f"context_dim={self.context_dim}")


# # Optional: Factory functions for convenience
# def create_resnet_block(
#     in_channels: int, 
#     out_channels: Optional[int] = None,
#     dropout: float = 0.1
# ) -> ResNetBlock:
#     """Factory function to create a ResNetBlock with default parameters."""
#     return ResNetBlock(
#         in_channels=in_channels,
#         out_channels=out_channels,
#         dropout=dropout
#     )


# def create_downsample_block(
#     in_channels: int,
#     scale_factor: float = 0.5,
#     mode: str = "nearest"
# ) -> Downsample:
#     """Factory function to create a Downsample block."""
#     return Downsample(
#         in_channels=in_channels,
#         scale_factor=scale_factor,
#         mode=mode
#     )


# def create_upsample_block(
#     in_channels: int,
#     scale_factor: float = 2.0,
#     mode: str = "nearest"
# ) -> Upsample:
#     """Factory function to create an Upsample block."""
#     return Upsample(
#         in_channels=in_channels,
#         scale_factor=scale_factor,
#         mode=mode
#     )


# def create_feedforward(
#     in_channels: int,
#     hidden_channels: int,
#     out_channels: Optional[int] = None,
#     dropout: float = 0.2
# ) -> FeedForward:
#     """Factory function to create a FeedForward block."""
#     return FeedForward(
#         in_channels=in_channels,
#         hidden_channels=hidden_channels,
#         out_channels=out_channels,
#         dropout=dropout
#     )


# # Optional: Test function to verify implementations
# def test_blocks() -> None:
#     """Test function to verify all blocks work correctly."""
#     import warnings
    
#     print("Testing blocks...")
    
#     # Test ResNetBlock
#     block = ResNetBlock(in_channels=64, out_channels=128, dropout=0.1)
#     x = torch.randn(4, 64, 32, 32)
#     y = block(x)
#     assert y.shape == (4, 128, 32, 32), f"ResNetBlock shape mismatch: {y.shape}"
#     print("✓ ResNetBlock passed")
    
#     # Test Downsample
#     downsample = Downsample(in_channels=64)
#     x = torch.randn(4, 64, 32, 32)
#     y = downsample(x)
#     assert y.shape == (4, 64, 16, 16), f"Downsample shape mismatch: {y.shape}"
#     print("✓ Downsample passed")
    
#     # Test Upsample
#     upsample = Upsample(in_channels=64)
#     x = torch.randn(4, 64, 16, 16)
#     y = upsample(x)
#     assert y.shape == (4, 64, 32, 32), f"Upsample shape mismatch: {y.shape}"
#     print("✓ Upsample passed")
    
#     # Test FeedForward
#     ff = FeedForward(in_channels=512, hidden_channels=2048, dropout=0.1)
#     x = torch.randn(4, 512)
#     y = ff(x)
#     assert y.shape == (4, 512), f"FeedForward shape mismatch: {y.shape}"
#     print("✓ FeedForward passed")
    
#     print("\nAll tests passed! ✅")


# if __name__ == "__main__":
#     # Run tests if this file is executed directly
#     test_blocks()