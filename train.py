import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from config import get_config
from dataset.dataset_modelnet import get_dataloader_modelnet
from agent import Agent
from utils.utils import acc
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.profilers import PyTorchProfiler


class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.agent = Agent(config, self.device)
        self.acc_list = [i for i in range(5, 35, 5)]

    def forward(self, x):
        return self.agent(x)

    def training_step(self, batch, _):
        rotation, ldjs, feature, A = self.agent(batch)
        result_dict = self.agent.compute_loss(rotation, ldjs, feature, A)
        self.log('train_loss', result_dict['loss'])
        return result_dict['loss']

    def validation_step(self, batch, _):
        rotation, ldjs, feature, A = self.agent(batch)
        result_dict = self.agent.compute_loss(rotation, ldjs, feature, A)
        self.log('val_loss', result_dict['loss'])
        return result_dict['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer


class ModelNetDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset, self.category_datasets, self.categories = get_dataloader_modelnet('train', self.config)
            self.val_dataset, _, _ = get_dataloader_modelnet('test', self.config)

        if stage == 'test' or stage is None:
            self.test_dataset, _, _ = get_dataloader_modelnet('test', self.config)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers, pin_memory=True)

def main():
    # Create experiment config containing all hyperparameters
    config = get_config("train")
    data_module = ModelNetDataModule(config)
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

    profiler = PyTorchProfiler(emit_nvtx=True) if config.profile else None
    
    # Create trainer
    trainer = Trainer(max_epochs=config.max_iteration, logger=logger, callbacks=[checkpoint_callback],
                      accelerator='gpu', devices=-1, profiler=profiler)
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()