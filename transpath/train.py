import os
import argparse
import yaml
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from .transpath import TransPathModel
from .lit_model import LitTransPath
from .datamodule import GridDataModule


def parse_args():
    parser = argparse.ArgumentParser(description='Train TransPath model')
    
    # Data
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--mode', type=str, default='h', choices=['h', 'cf', 'f', 'nastar'])
    
    # Model
    parser.add_argument('--model_config', type=str, default='configs/model.yaml')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--out_channels', type=int, default=1)
    
    # Training
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--clip_value', type=float, default=0.95)
    
    # Data module
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    
    # Trainer
    parser.add_argument('--accelerator', type=str, default='auto', choices=['cpu', 'gpu', 'tpu', 'auto'])
    parser.add_argument('--devices', type=int, default=1)
    
    # Logging
    parser.add_argument('--exp_name', type=str, default='transpath')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    
    # Callbacks
    parser.add_argument('--monitor', type=str, default='val_loss')
    parser.add_argument('--mode_min', type=str, default='min', choices=['min', 'max'])
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_top_k', type=int, default=3)
    
    return parser.parse_args()


def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def setup_components(args):
    """Setup all components in one function"""
    # Directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Model config
    model_config = load_config(args.model_config)
    
    # Data
    datamodule = GridDataModule(
        data_path=args.data_path,
        mode=args.mode,
        clip_value=args.clip_value,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        shuffle_train=True,
    )
    
    # Model
    model = TransPathModel(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        **model_config
    )
    
    lit_model = LitTransPath(
        model=model,
        mode=args.mode,
        learning_rate=args.lr,
        weight_decay=args.weight_decay

    )
    
    # Logger
    logger = TensorBoardLogger(args.log_dir, name=args.exp_name)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.save_dir, args.exp_name),
            filename='{epoch:03d}-{val_loss:.4f}',
            monitor=args.monitor,
            mode=args.mode_min,
            save_top_k=args.save_top_k,
            save_last=True
        ),
        EarlyStopping(
            monitor=args.monitor,
            mode=args.mode_min,
            patience=args.patience
        ),
        LearningRateMonitor()
    ]
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=args.save_dir,
        log_every_n_steps=1,
        deterministic=False,
        enable_progress_bar=True,   # ← явно включить (по умолчанию True)
        limit_train_batches=10,   # обучение: только первые 10 батчей
        limit_val_batches=10,     # валидация: только первые 10 батчей
    )
    
    return datamodule, lit_model, trainer, logger


def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    
    print(f"Training {args.exp_name} in {args.mode} mode")
    print(f"Data: {args.data_path}, Batch: {args.batch_size}, LR: {args.lr}")
    
    try:
        datamodule, model, trainer, logger = setup_components(args)
        
        # Log hyperparams
        logger.log_hyperparams(vars(args))
        
        print("Starting training...")
        # Train
        trainer.fit(model, datamodule=datamodule)
        
        # Test
        test_results = trainer.test(model, datamodule=datamodule, ckpt_path='best')
        
        # Save final model
        checkpoint_dir = os.path.join(args.save_dir, args.exp_name)
        final_path = os.path.join(checkpoint_dir, "model_final.pt")
        torch.save({
            'model_state_dict': model.model.state_dict(),
            'config': model.model.get_config(),
            'args': vars(args)
        }, final_path)
        
        print(f"\nBest model: {trainer.checkpoint_callback.best_model_path}")
        print(f"Final model: {final_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()