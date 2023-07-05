import os
from pathlib import Path
from typing import Tuple

import lightning
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import torch
from torch.utils.data import Dataset as _Dataset, random_split
from torch.utils.data.dataloader import DataLoader

from agenc.data import Dataset
from agenc.learner import Learner


class TorchDataset(_Dataset):
    def __init__(self, data: Dataset):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, outputs = self.data[item]
        return torch.Tensor(inputs), torch.Tensor(outputs)


class SimpleDense(Learner):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        num_workers: int = os.cpu_count() or 1,
        max_epochs: int = 100,
    ):
        self.model = MultilayerPerceptron(learning_rate=learning_rate)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs

    def train(self, dataset: Dataset):
        train_len = int(0.8 * len(dataset))
        validation_len = len(dataset) - train_len
        train_dataset, val_dataset = random_split(
            TorchDataset(dataset),
            [train_len, validation_len],
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        validation_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        early_stopping = EarlyStopping(
            monitor="validate/loss",
            patience=5,
            mode="min",
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="validate/loss",
            mode="min",
            save_last=True,
        )
        logger = TensorBoardLogger(Path.cwd(), default_hp_metric=False)
        trainer = lightning.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[early_stopping, checkpoint_callback],
            logger=logger,
            enable_checkpointing=True,
        )
        trainer.fit(self.model, train_dataloader, validation_dataloader)

        self.model = MultilayerPerceptron.load_from_checkpoint(
            checkpoint_callback.best_model_path,
        )

    def predict(self, dataset: Dataset) -> np.ndarray:
        dataloader = DataLoader(
            TorchDataset(dataset),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        predictions = []
        for batch in dataloader:
            inputs, _ = batch
            predictions.append(self.model(inputs).detach().numpy())
        return np.concatenate(predictions)


class MultilayerPerceptron(lightning.LightningModule):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = torch.nn.Sequential(
            torch.nn.Linear(3, 25),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(25, 25),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(25, 1),
        )

    def forward(self, x):
        input_mean = torch.Tensor([0.12495715, 0.10395051, 0.02667484])
        input_std = torch.Tensor([0.17149029, 0.11083332, 0.01697188])
        output_mean = torch.Tensor([3725.85228508])
        output_std = torch.Tensor([3710.73972826])
        return self.model((x - input_mean) / input_std) * output_std + output_mean

    def _shared_eval_step(self, batch):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        return loss

    def training_step(self, batch, _):
        loss = self._shared_eval_step(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, _):
        loss = self._shared_eval_step(batch)
        self.log("validate/loss", loss)
        return loss

    def test_step(self, batch, _):
        loss = self._shared_eval_step(batch)
        self.log("test/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=40,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
