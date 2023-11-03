import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from config import get_config
from dataset import get_dataloader
from agent import Agent
from utils.utils import acc
import torch
import numpy as np
import random

from pytorch_lightning.profilers import PyTorchProfiler

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.agent = Agent(config, self.device)
        self.acc_list = [i for i in range(5, 35, 5)]

    def forward(self, x):
        return self.agent(x)

    def train_dataloader(self):
        train_loader, _, _ = get_dataloader(self.config.dataset, "train", self.config)
        return train_loader

    def val_dataloader(self):
        val_loader, _, _ = get_dataloader(self.config.dataset, "test", self.config)
        return val_loader

    def training_step(self, batch, batch_idx):
        rotation, ldjs, feature, A = self.agent(batch)
        result_dict = self.agent.compute_loss(rotation, ldjs, feature, A)
        self.log('train_loss', result_dict['loss'])
        return result_dict['loss']

    def validation_step(self, batch, batch_idx):
        rotation, ldjs, feature, A = self.agent(batch)
        result_dict = self.agent.compute_loss(rotation, ldjs, feature, A)
        self.log('val_loss', result_dict['loss'])
        return result_dict['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer

def main():
    # Create experiment config containing all hyperparameters
    config = get_config("train")

    # create network and training agent
    model = LitModel(config)

    # Create logger
    logger = TensorBoardLogger('tb_logs', name='my_model')

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='my_model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    profile = True
    if profile:
        profiler = PyTorchProfiler(emit_nvtx=True)
    else:
        profiler = None
    # Create trainer
    trainer = Trainer(max_epochs=config.max_iteration, logger=logger, callbacks=[checkpoint_callback], gpus=-1, profiler=profiler)
    trainer.fit(model)

if __name__ == "__main__":
    main()