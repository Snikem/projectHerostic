from einops import rearrange
from typing import Optional
import torch.nn as nn
import torch

from .blocks import BasicTransformerBlock, FeedForward  # Import from blocks


class SpatialTransformer(nn.Module):
    """
    Spatial Transformer for image-like data.
    
    Adapts Transformer blocks to process 2D spatial data by:
    1. Applying GroupNorm to the input
    2. Reshaping from (B, C, H, W) to (B, H*W, C)
    3. Processing through multiple Transformer blocks
    4. Reshaping back to (B, C, H, W)
    5. Adding residual connection to original input
    
    This enables attention mechanisms to operate on spatial positions
    as if they were tokens in a sequence.
    
    Args:
        in_channels: Number of input channels (embedding dimension)
        num_heads: Number of attention heads
        depth: Number of Transformer blocks
        dropout: Dropout probability
        context_dim: Dimension of context for cross-attention
        
    Attributes:
        in_channels: Number of input channels
        norm: GroupNorm applied before Transformer
        transformer_blocks: ModuleList of BasicTransformerBlocks
        
    Examples:
        >>> transformer = SpatialTransformer(in_channels=64, num_heads=8, depth=4)
        >>> x = torch.randn(2, 64, 32, 32)  # (batch, channels, height, width)
        >>> output = transformer(x)
        >>> output.shape
        torch.Size([2, 64, 32, 32])
        
        >>> # With cross-attention context
        >>> context = torch.randn(2, 16, 128)  # (batch, context_seq_len, context_dim)
        >>> output = transformer(x, context=context)
        >>> output.shape
        torch.Size([2, 64, 32, 32])
    """
    
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        depth: int = 4,
        dropout: float = 0.3,
        context_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        
        # Normalization before Transformer
        self.norm = nn.GroupNorm(
            num_groups=32,
            num_channels=in_channels,
            eps=1e-6,
            affine=True
        )
        
        # Stack of Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                embed_dim=in_channels,
                num_heads=num_heads,
                dropout=dropout,
                context_dim=context_dim
            )
            for _ in range(depth)
        ])

    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the Spatial Transformer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            context: Optional context tensor for cross-attention of shape
                    (batch_size, context_length, context_dim). If None,
                    each block uses self-attention.
                    
        Returns:
            Output tensor of shape (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.shape
        
        # Store original for residual connection
        residual = x
        
        # Apply normalization
        f = self.norm(x)
        
        # Reshape: (B, C, H, W) -> (B, H*W, C)
        f = rearrange(f, 'b c h w -> b (h w) c')
        
        # Process through Transformer blocks
        for block in self.transformer_blocks:
            f = block(f, context=context)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        f = rearrange(f, 'b (h w) c -> b c h w', h=height, w=width)
        
        # Add residual connection
        return f + residual
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f"in_channels={self.in_channels}, "
                f"num_heads={self.num_heads}, "
                f"depth={len(self.transformer_blocks)}")


# # Тестирующая функция
# def test_spatial_transformer():
#     """Test the SpatialTransformer module."""
#     print("Testing SpatialTransformer...")
    
#     # Тест 1: Без контекста
#     transformer = SpatialTransformer(in_channels=64, num_heads=8, depth=2)
#     x = torch.randn(2, 64, 32, 32)
#     output = transformer(x)
    
#     assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
#     assert not torch.allclose(output, x), "Output should be different from input"
#     print("✓ Test 1 passed: Basic forward pass")
    
#     # Тест 2: С контекстом
#     transformer2 = SpatialTransformer(in_channels=64, num_heads=8, depth=2, context_dim=128)
#     x = torch.randn(2, 64, 32, 32)
#     context = torch.randn(2, 16, 128)  # (batch, context_seq_len, context_dim)
#     output = transformer2(x, context=context)
    
#     assert output.shape == x.shape, f"Shape mismatch with context: {output.shape}"
#     print("✓ Test 2 passed: With context")
    
#     # Тест 3: Градиенты
#     x = torch.randn(1, 64, 16, 16, requires_grad=True)
#     transformer = SpatialTransformer(in_channels=64, num_heads=4, depth=1)
#     output = transformer(x)
#     loss = output.sum()
#     loss.backward()
    
#     assert x.grad is not None, "Gradients should be computed"
#     print("✓ Test 3 passed: Gradient computation")
    
#     # Тест 4: Разные размеры
#     transformer = SpatialTransformer(in_channels=32, num_heads=4, depth=3)
#     for h, w in [(8, 8), (16, 16), (32, 32), (64, 64)]:
#         x = torch.randn(1, 32, h, w)
#         output = transformer(x)
#         assert output.shape == (1, 32, h, w), f"Failed for size {h}x{w}"
#     print("✓ Test 4 passed: Different spatial sizes")
    
#     print("\n✅ All SpatialTransformer tests passed!")


# if __name__ == "__main__":
#     # Запуск тестов если файл выполняется напрямую
#     test_spatial_transformer()