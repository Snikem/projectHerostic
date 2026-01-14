import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from .blocks import ResNetBlock, Downsample, Upsample
from .pos_embeddings import PosEmbeds


class Encoder(nn.Module):
    """
    Encoder module for downsampling spatial features.
    
    The encoder progressively reduces spatial dimensions while increasing
    feature complexity through a series of convolutional and downsampling blocks.
    
    Architecture:
        Input -> Conv2d -> [ResNetBlock -> Downsample] * N
        
    Args:
        in_channels: Number of input channels
        hidden_channels: Number of hidden/feature channels
        downsample_steps: Number of downsampling steps (each reduces spatial size by 2)
        dropout: Dropout probability for ResNet blocks
        
    Attributes:
        layers: ModuleList containing all encoder layers
        in_channels: Number of input channels
        hidden_channels: Number of hidden channels
        downsample_steps: Number of downsampling steps
        
    Examples:
        >>> encoder = Encoder(in_channels=2, hidden_channels=64, downsample_steps=3)
        >>> x = torch.randn(4, 2, 64, 64)
        >>> output = encoder(x)
        >>> output.shape  # After 3 downsampling steps: 64 -> 32 -> 16 -> 8
        torch.Size([4, 64, 8, 8])
        
    Note:
        Each downsampling step reduces spatial dimensions by factor of 2.
        Total downsampling factor = 2^downsample_steps
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        downsample_steps: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.downsample_steps = downsample_steps
        
        # Initial convolution to project to hidden dimension
        self.layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=5,
                stride=1,
                padding=2
            )
        ])
        
        # Add downsampling blocks
        for i in range(downsample_steps):
            self.layers.append(
                nn.Sequential(
                    ResNetBlock(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        dropout=dropout
                    ),
                    Downsample(in_channels=hidden_channels)
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Encoded features of shape 
            (batch_size, hidden_channels, height/2^N, width/2^N)
            where N = downsample_steps
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Calculate output shape given input shape.
        
        Args:
            input_shape: Tuple of (channels, height, width)
            
        Returns:
            Tuple of (hidden_channels, height_out, width_out)
        """
        channels, height, width = input_shape
        
        # Calculate spatial dimensions after downsampling
        height_out = height // (2 ** self.downsample_steps)
        width_out = width // (2 ** self.downsample_steps)
        
        return (self.hidden_channels, height_out, width_out)
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f"in_channels={self.in_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"downsample_steps={self.downsample_steps}")


class Decoder(nn.Module):
    """
    Decoder module for upsampling features to original spatial resolution.
    
    The decoder progressively increases spatial dimensions while refining
    features through a series of upsampling and ResNet blocks.
    
    Architecture:
        Input -> [ResNetBlock -> Upsample] * N -> GroupNorm -> SiLU -> Conv2d -> Tanh
        
    Args:
        hidden_channels: Number of input/hidden channels
        out_channels: Number of output channels
        upsample_steps: Number of upsampling steps (each increases spatial size by 2)
        dropout: Dropout probability for ResNet blocks
        
    Attributes:
        layers: ModuleList containing upsampling blocks
        norm: Group normalization layer
        silu: SiLU activation function
        conv_out: Final convolutional layer
        hidden_channels: Number of hidden channels
        out_channels: Number of output channels
        upsample_steps: Number of upsampling steps
        
    Examples:
        >>> decoder = Decoder(hidden_channels=64, out_channels=1, upsample_steps=3)
        >>> x = torch.randn(4, 64, 8, 8)
        >>> output = decoder(x)
        >>> output.shape  # After 3 upsampling steps: 8 -> 16 -> 32 -> 64
        torch.Size([4, 1, 64, 64])
        >>> output.min(), output.max()  # Tanh activation ensures output in [-1, 1]
        (tensor(-1.), tensor(1.))
    """
    
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        upsample_steps: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.upsample_steps = upsample_steps
        
        # Upsampling blocks
        self.layers = nn.ModuleList([])
        for i in range(upsample_steps):
            self.layers.append(
                nn.Sequential(
                    ResNetBlock(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        dropout=dropout
                    ),
                    Upsample(in_channels=hidden_channels)
                )
            )
        
        # Final layers
        self.norm = nn.GroupNorm(
            num_groups=32,
            num_channels=hidden_channels,
            eps=1e-6,
            affine=True
        )
        self.silu = nn.SiLU()
        self.conv_out = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_channels, height, width)
            
        Returns:
            Decoded output of shape 
            (batch_size, out_channels, height*2^N, width*2^N)
            where N = upsample_steps
            Output values are in range [-1, 1] due to Tanh activation
        """
        # Apply upsampling blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final processing
        x = self.norm(x)
        x = self.silu(x)
        x = self.conv_out(x)
        
        # Tanh activation ensures output is in [-1, 1]
        return torch.tanh(x)
    
    def get_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Calculate output shape given input shape.
        
        Args:
            input_shape: Tuple of (channels, height, width)
            
        Returns:
            Tuple of (out_channels, height_out, width_out)
        """
        channels, height, width = input_shape
        
        # Calculate spatial dimensions after upsampling
        height_out = height * (2 ** self.upsample_steps)
        width_out = width * (2 ** self.upsample_steps)
        
        return (self.out_channels, height_out, width_out)
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f"hidden_channels={self.hidden_channels}, "
                f"out_channels={self.out_channels}, "
                f"upsample_steps={self.upsample_steps}")




# # Test functions
# def test_encoder_decoder():
#     """Test Encoder and Decoder modules."""
#     print("Testing Encoder and Decoder...")
    
#     # Test Encoder
#     encoder = Encoder(in_channels=2, hidden_channels=64, downsample_steps=3)
#     x = torch.randn(4, 2, 64, 64)
#     encoded = encoder(x)
    
#     expected_shape = (4, 64, 8, 8)  # 64 / 2^3 = 8
#     assert encoded.shape == expected_shape, f"Encoder shape mismatch: {encoded.shape}"
#     print("✓ Encoder passed")
    
#     # Test Decoder
#     decoder = Decoder(hidden_channels=64, out_channels=1, upsample_steps=3)
#     decoded = decoder(encoded)
    
#     expected_shape = (4, 1, 64, 64)  # 8 * 2^3 = 64
#     assert decoded.shape == expected_shape, f"Decoder shape mismatch: {decoded.shape}"
    
#     # Check Tanh output range
#     assert decoded.min() >= -1.0 and decoded.max() <= 1.0, "Decoder output not in [-1, 1]"
#     print("✓ Decoder passed")
    
#     # Test PosEmbeds
#     pos_embeds = PosEmbeds(hidden_size=64, resolution=(32, 32))
#     features = torch.randn(2, 64, 32, 32)
#     output = pos_embeds(features)
    
#     assert output.shape == features.shape, f"PosEmbeds shape mismatch: {output.shape}"
#     assert not torch.allclose(output, features), "PosEmbeds should modify input"
#     print("✓ PosEmbeds passed")
    
#     # Test encoder-decoder round trip (not identity due to down/upsampling)
#     encoder = Encoder(in_channels=3, hidden_channels=32, downsample_steps=2)
#     decoder = Decoder(hidden_channels=32, out_channels=3, upsample_steps=2)
    
#     x = torch.randn(2, 3, 32, 32)
#     encoded = encoder(x)
#     decoded = decoder(encoded)
    
#     assert decoded.shape == x.shape, f"Round trip shape mismatch: {decoded.shape}"
#     print("✓ Encoder-Decoder round trip passed")
    
#     print("\n✅ All Encoder/Decoder tests passed!")


# if __name__ == "__main__":
#     test_encoder_decoder()