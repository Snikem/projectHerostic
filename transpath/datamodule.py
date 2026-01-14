from typing import Optional, Dict, Any
import torch
from torch.utils.data import DataLoader, random_split
import lightning as L

from .data_prep import GridDataset


class GridDataModule(L.LightningDataModule):
    """
    LightningDataModule for path planning tasks.
    
    Manages data loading, splitting, and preparation for training.
    
    Args:
        data_path: Path to data directory
        mode: Data mode ('f', 'nastar', 'h', 'cf')
        clip_value: Clip value for 'f' mode
        batch_size: Batch size
        num_workers: Number of data loader workers
        val_split: Validation split ratio
        test_split: Test split ratio
        seed: Random seed for reproducibility
        shuffle_train: Whether to shuffle training data
    """
    
    def __init__(
        self,
        data_path: str,
        mode: str = 'f',
        clip_value: float = 0.95,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        shuffle_train: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['kwargs'])
        
        # Store parameters
        self.data_path = data_path
        self.mode = mode
        self.clip_value = clip_value
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.shuffle_train = shuffle_train
        
        # Datasets (will be initialized in setup)
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Statistics
        self.stats = None

    def prepare_data(self) -> None:
        """
        Download or verify data existence.
        Called only once per node.
        """
        # Check if data files exist
        required_files = ['maps.npy', 'goals.npy', 'starts.npy']
        
        if self.mode == 'f':
            required_files.append('focal.npy')
        elif self.mode == 'h' or self.mode == 'nastar':
            required_files.append('abs.npy')
        elif self.mode == 'cf':
            required_files.append('cf.npy')
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        import os
        for file in required_files:
            file_path = os.path.join(self.data_path, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Data file not found: {file_path}\n"
                    f"Please ensure data is available at: {self.data_path}"
                )
        
        print(f"Data verified for mode '{self.mode}'")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets for each stage.
        
        Args:
            stage: Either 'fit', 'validate' or 'test'
        """
        # Initialize full dataset
        if self.full_dataset is None:
            self.full_dataset = GridDataset(
                path=self.data_path,
                mode=self.mode,
                clip_value=self.clip_value
            )
            
            # Get dataset statistics
            print(f"Dataset statistics: {self.stats}")
        
        # Split dataset if not already split
        if stage in ('fit', None) and self.train_dataset is None:
            self._split_datasets()
    
    def _split_datasets(self) -> None:
        """Split dataset into train/val/test."""
        total_size = len(self.full_dataset)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size - test_size
        
        # Set random seed for reproducibility
        generator = torch.Generator().manual_seed(self.seed)
        
        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset,
            [train_size, val_size, test_size],
            generator=generator
        )
        
        print(f"Dataset split:")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Val:   {len(self.val_dataset)} samples")
        print(f"  Test:  {len(self.test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        """
        Create training DataLoader.
        
        Returns:
            DataLoader for training
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True  # Drop last incomplete batch
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create validation DataLoader.
        
        Returns:
            DataLoader for validation
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create test DataLoader.
        
        Returns:
            DataLoader for testing
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Get DataModule configuration.
        
        Returns:
            Dictionary with configuration
        """
        config = {
            'data_path': self.data_path,
            'mode': self.mode,
            'clip_value': self.clip_value,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'val_split': self.val_split,
            'test_split': self.test_split,
            'seed': self.seed,
            'shuffle_train': self.shuffle_train,
        }
        
        if self.stats:
            config.update({
                'dataset_size': len(self.full_dataset) if self.full_dataset else 0,
                'stats': self.stats
            })
        
        return config