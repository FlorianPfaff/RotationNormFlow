import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from config import get_config
from dataset.dataset_modelnet import PrecomputedModelNetDataModule, ModelNetDataModule
from agent import Agent, PrecomputedFeaturesAgent
import torch

from pytorch_lightning.profilers import PyTorchProfiler


class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.use_precomputed_features:
            self.agent = PrecomputedFeaturesAgent(config)
        else:
            self.agent = Agent(config)
            
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


def main():
    # Create experiment config containing all hyperparameters
    config = get_config("train")
    if config.use_precomputed_features:
        data_module = PrecomputedModelNetDataModule(
            features_paths={'train': 'train_features.npz', 'val': 'val_features.npz', 'test': 'test_features.npz'},
            metadata_paths={'train': 'train_metadata.pkl', 'val': 'val_metadata.pkl', 'test': 'test_metadata.pkl'},
            batch_size=config.batch_size
        )
    else:
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
    trainer = pl.Trainer(max_epochs=config.max_iteration, logger=logger, callbacks=[checkpoint_callback],
                      accelerator='gpu', devices=-1, profiler=profiler)
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()