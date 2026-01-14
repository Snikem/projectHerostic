import torch
import torch.nn as nn
import lightning as L
from torch import Tensor
from typing import Tuple, Dict, Any, Optional


class LitTransPath(L.LightningModule):
    """
    LightningModule для обучения TransPathModel.
    
    Обертка над TransPathModel для обучения с PyTorch Lightning.
    Поддерживает различные режимы обучения (эвристика, планирование пути).
    
    Args:
        model: Экземпляр TransPathModel
        mode: Режим работы ('h' - эвристика, 'f' - планирование пути, 'nastar' - NA*)
        learning_rate: Скорость обучения
        weight_decay: Вес decay для регуляризации L2
        use_scheduler: Использовать ли планировщик скорости обучения
        scheduler_type: Тип планировщика ('onecycle', 'cosine', 'reduce_on_plateau')
        
    Attributes:
        model: TransPath модель
        mode: Режим работы
        loss: Функция потерь
        k: Масштабирующий коэффициент для потерь
        learning_rate: Скорость обучения
        weight_decay: Вес decay
        use_scheduler: Использовать планировщик
        scheduler_type: Тип планировщика
        
    Examples:
        >>> model = TransPathModel(in_channels=3, out_channels=1)
        >>> lit_model = TransPathLit(model, mode='h', learning_rate=1e-4)
        >>> trainer = L.Trainer(max_epochs=10)
        >>> trainer.fit(lit_model, train_dataloader, val_dataloader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        mode: str = 'h',
        learning_rate: float = 4e-4,
        weight_decay: float = 0.0,
        scheduler_type: str = 'onecycle',
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        # Сохраняем модель и параметры
        self.model = model
        self.mode = mode
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        
        # Выбираем функцию потерь в зависимости от режима
        if mode == 'h':
            self.loss = nn.L1Loss()  # MAE для эвристических карт
            self.k = 64 * 64  # Масштабирование для карт эвристик
        else:  # 'f', 'cf' или 'nastar'
            self.loss = nn.MSELoss()  # MSE для карт планирования
            self.k = 1  # Без масштабирования
        
        # Метрики для мониторинга
        self.train_loss = []
        self.val_loss = []
        
        # Примеры входов для автоматической оптимизации графа
        self.example_input_array = torch.randn(1, 2, 64, 64)

    def forward(self, x: Tensor) -> Tensor:
        """
        Прямой проход через модель.
        
        Args:
            x: Входной тензор (batch_size, channels, height, width)
            
        Returns:
            Выходной тензор (batch_size, out_channels, height, width)
        """
        return self.model(x)

    def prepare_inputs(
        self,
        map_design: Tensor,
        start: Tensor,
        goal: Tensor,
        mode: Optional[str] = None
    ) -> Tensor:
        """
        Подготовка входных данных для модели в зависимости от режима.
        
        Args:
            map_design: Тензор карты (batch_size, 1, H, W)
            start: Тензор стартовой позиции (batch_size, 1, H, W)
            goal: Тензор целевой позиции (batch_size, 1, H, W)
            mode: Режим (если None, используется self.mode)
            
        Returns:
            Конкатенированный входной тензор
        """
        map_design = map_design.squeeze(2)
        start = start.squeeze(2)
        goal = goal.squeeze(2)

        mode = mode or self.mode

        if mode in ("h", "cf"):
            return torch.cat([map_design, goal], dim=1)
        else:
            return torch.cat([map_design, start + goal], dim=1)

    def scale_predictions(self, predictions: Tensor) -> Tensor:
        """
        Масштабирование предсказаний модели.
        
        Модель выдает значения в диапазоне [-1, 1] (из-за tanh в декодере).
        Масштабируем их к диапазону [0, k].
        
        Args:
            predictions: Предсказания модели в диапазоне [-1, 1]
            
        Returns:
            Масштабированные предсказания в диапазоне [0, k]
        """
        return (predictions + 1) / 2 * self.k

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor],
        batch_idx: int
    ) -> Dict[str, Tensor]:
        """
        Шаг обучения.
        
        Args:
            batch: Кортеж (map_design, start, goal, gt_hmap)
            batch_idx: Индекс батча
            
        Returns:
            Словарь с loss и логируемыми значениями
        """
        map_design, start, goal, gt_hmap = batch
        
        # Подготавливаем входные данные
        inputs = self.prepare_inputs(map_design, start, goal)
        
        # Прямой проход
        predictions = self.model(inputs)
        
        # Масштабируем предсказания
        scaled_preds = self.scale_predictions(predictions)
        
        # Вычисляем потери
        loss = self.loss(scaled_preds, gt_hmap)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Логируем метрики
        self.log('train_loss', loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 batch_size=inputs.size(0))

        
        return {'loss': loss, 'preds': scaled_preds, 'targets': gt_hmap}

    # def on_train_epoch_end(self) -> None:
    #     """Вызывается в конце каждой эпохи обучения."""
    #     if self.train_loss:
    #         avg_loss = torch.stack(self.train_loss).mean()
    #         self.log('train_loss_epoch', avg_loss, prog_bar=True, logger=True)
    #         self.train_loss.clear()

    def validation_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor],
        batch_idx: int
    ) -> Optional[Dict[str, Tensor]]:
        """
        Шаг валидации.
        
        Args:
            batch: Кортеж (map_design, start, goal, gt_hmap)
            batch_idx: Индекс батча
            
        Returns:
            Словарь с loss и предсказаниями (опционально)
        """
        map_design, start, goal, gt_hmap = batch
        
        # Подготавливаем входные данные
        inputs = self.prepare_inputs(map_design, start, goal)
        
        # Прямой проход
        predictions = self.model(inputs)
        
        # Масштабируем предсказания
        scaled_preds = self.scale_predictions(predictions)
        
        # Вычисляем потери
        loss = self.loss(scaled_preds, gt_hmap)
        
        # Логируем метрики
        self.log('val_loss', loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 batch_size=inputs.size(0),
                 sync_dist=True)
        
        # Сохраняем для усреднения

        
        # Можно вернуть для дополнительной обработки
        return {'val_loss': loss, 'preds': scaled_preds, 'targets': gt_hmap}

    # def on_validation_epoch_end(self) -> None:
    #     """Вызывается в конце каждой эпохи валидации."""
    #     if self.val_loss:
    #         avg_loss = torch.stack(self.val_loss).mean()
    #         self.log('val_loss_epoch', avg_loss, prog_bar=True, logger=True)
    #         self.val_loss.clear()

    def test_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor],
        batch_idx: int
    ) -> Optional[Dict[str, Tensor]]:
        """
        Шаг тестирования.
        
        Args:
            batch: Кортеж (map_design, start, goal, gt_hmap)
            batch_idx: Индекс батча
            
        Returns:
            Словарь с loss и предсказаниями
        """
        map_design, start, goal, gt_hmap = batch
        
        # Подготавливаем входные данные
        inputs = self.prepare_inputs(map_design, start, goal)
        
        # Прямой проход
        predictions = self.model(inputs)
        
        # Масштабируем предсказания
        scaled_preds = self.scale_predictions(predictions)
        
        # Вычисляем потери
        loss = self.loss(scaled_preds, gt_hmap)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Дополнительные метрики
        mae = nn.L1Loss()(scaled_preds, gt_hmap)
        mse = nn.MSELoss()(scaled_preds, gt_hmap)
        
        # Логируем метрики
        self.log_dict({
            'test_loss': loss,
            'test_mae': mae,
            'test_mse': mse
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {
            'test_loss': loss,
            'preds': scaled_preds,
            'targets': gt_hmap,
            'inputs': inputs
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Настройка оптимизаторов и планировщиков.
        
        Returns:
            Словарь с оптимизатором и планировщиками
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        if self.scheduler_type == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=10000.0,
                anneal_strategy='cos'
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
            
        elif self.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,  # Количество эпох до первого рестарта
                T_mult=2,  # Умножение T_0 после каждого рестарта
                eta_min=1e-6  # Минимальная скорость обучения
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
            
        elif self.scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True,
                min_lr=1e-6
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }

        else:
            return {
                'optimizer': optimizer
            }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Вызывается при сохранении checkpoint."""
        # Добавляем информацию о модели
        checkpoint['model_config'] = self.model.get_config()
        checkpoint['lit_config'] = {
            'mode': self.mode,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Вызывается при загрузке checkpoint."""
        if 'model_config' in checkpoint:
            self.logger.info(f"Loaded model config: {checkpoint['model_config']}")