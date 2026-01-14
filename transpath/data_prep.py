import os
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class GridDataset(Dataset):
    """
    Pure PyTorch Dataset for path planning data.

    Args:
        path: Path to data directory
        mode: Data mode ('f', 'h', 'cf')
        clip_value: Clip value for 'f' mode
    """
    
    def __init__(
        self,
        path: str,
        mode: str = 'f',
        clip_value: float = 0.95,
    ):
        self.mode = mode
        self.clip_value = clip_value

        # Load data
        self.maps = np.load(os.path.join(path, 'maps.npy'), mmap_mode='c')
        self.starts = np.load(os.path.join(path, 'starts.npy'), mmap_mode='c')
        self.goals = np.load(os.path.join(path, 'goals.npy'), mmap_mode='c')

        
        # Select ground truth file based on mode
        file_gt = {'f': 'focal.npy', 'h': 'abs.npy', 'cf': 'cf.npy'}[mode]
        self.gt_values = np.load(os.path.join(path, file_gt), mmap_mode='c')
      
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, 
                                             torch.Tensor, torch.Tensor]:
        # Load data
        map_data = torch.from_numpy(self.maps[idx].astype('float32'))
        start = torch.from_numpy(self.starts[idx].astype('float32'))
        goal = torch.from_numpy(self.goals[idx].astype('float32'))
        gt = torch.from_numpy(self.gt_values[idx].astype('float32'))
        
        # Apply clipping for 'f' mode
        if self.mode == 'f':
            gt = torch.where(gt >= self.clip_value, gt, torch.zeros_like(gt))
        
        # Add channel dimension
        map_data = map_data.unsqueeze(0)  # (1, H, W)
        start = start.unsqueeze(0)
        goal = goal.unsqueeze(0)
        gt = gt.unsqueeze(0)
        
        return map_data, start, goal, gt
    
    def __len__(self):
        # Возвращает общее количество образцов
        return len(self.maps)
