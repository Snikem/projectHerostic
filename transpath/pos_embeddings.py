import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class PosEmbeds(nn.Module):
    """
    Positional embeddings for spatial data.
    
    Adds learnable positional information to feature maps using a
    sinusoidal grid encoding.
    
    Args:
        hidden_size: Dimension of positional embeddings (should match feature dimension)
        resolution: Tuple of (height, width) for the grid
        max_v: Maximum value for grid coordinates (default: 1.0 for normalized coordinates)
        
    Attributes:
        linear: Linear layer to project grid to hidden dimension
        grid: Fixed positional grid (not trainable)
        resolution: Grid resolution
        max_v: Maximum coordinate value
        
    Examples:
        >>> pos_embeds = PosEmbeds(hidden_size=64, resolution=(32, 32))
        >>> features = torch.randn(2, 64, 32, 32)
        >>> output = pos_embeds(features)
        >>> output.shape
        torch.Size([2, 64, 32, 32])
    """
    
    def __init__(
        self,
        hidden_size: int,
        resolution: Tuple[int, int],
        max_v: float = 1.0
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.resolution = resolution
        self.max_v = max_v
        
        self.linear = nn.Linear(4, hidden_size)
        
        # Create fixed positional grid (not trainable)
        grid_tensor = torch.tensor(
            self._build_grid(resolution, max_v),
            dtype=torch.float32
        )
        self.register_buffer("grid", grid_tensor)

    @staticmethod
    def _build_grid(resolution: Tuple[int, int], max_v: float = 1.0) -> np.ndarray:
        """
        Build positional grid for embeddings.
        
        Args:
            resolution: Tuple of (height, width)
            max_v: Maximum coordinate value
            
        Returns:
            Grid array of shape (1, height, width, 4)
        """
        ranges = [np.linspace(0., max_v, num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        
        # Concatenate with complementary coordinates
        return np.concatenate([grid, max_v - grid], axis=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to input features.
        
        Args:
            inputs: Input tensor of shape (batch_size, hidden_size, height, width)
            
        Returns:
            Tensor with added positional embeddings, same shape as input
        """
        # Project grid to hidden dimension and reshape
        pos_emb = self.linear(self.grid)  # (1, H, W, hidden_size)
        pos_emb = pos_emb.permute(0, 3, 1, 2)  # (1, hidden_size, H, W)
        
        # Add to inputs (broadcast across batch dimension)
        return inputs + pos_emb
    
    def change_resolution(self, resolution: Tuple[int, int], max_v: float = 1.0) -> None:
        """
        Change the resolution of positional grid.
        
        Useful for variable resolution inputs or multi-scale processing.
        
        Args:
            resolution: New resolution tuple (height, width)
            max_v: New maximum coordinate value
        """
        self.resolution = resolution
        self.max_v = max_v
        
        # Create new grid
        new_grid = torch.tensor(
            self._build_grid(resolution, max_v),
            dtype=torch.float32,
            device=self.grid.device
        )
        
        # Update buffer
        self.grid = new_grid
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f"hidden_size={self.hidden_size}, "
                f"resolution={self.resolution}, "
                f"max_v={self.max_v}")