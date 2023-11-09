import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from config import get_config
from dataset.dataset_modelnet import PrecomputedModelNetDataModule, ModelNetDataModule
from agent import Agent, PrecomputedFeaturesAgent
import torch
import torch.optim as optim


from pytorch_lightning.profilers import PyTorchProfiler


class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        if config.use_precomputed_features:
            self.agent = PrecomputedFeaturesAgent(config)
        else:
            self.agent = Agent(config)
            
        self.acc_list = [i for i in range(5, 35, 5)]
        self.optimizer_flow = optim.Adam(self.agent.flow.parameters(), config.lr)

    def forward(self, x):
        return self.agent(x)

    def training_step(self, batch, _):
        rotation, ldjs, feature, A = self.agent(batch)
        result_dict = self.agent.compute_loss(rotation, ldjs, feature, A)
        self.log('loss', result_dict['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return result_dict['loss']

    def validation_step(self, batch, _):
        rotation, ldjs, feature, A = self.agent(batch)
        result_dict = self.agent.compute_loss(rotation, ldjs, feature, A)
        self.log('val_loss', result_dict['loss'])
        return result_dict['loss']

    """
    def configure_optimizers(self):
        if self.config.use_lr_decay:
            lr_decay = [int(item) for item in self.config.lr_decay.split(',')]
            optimizer_flow_scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer_flow, milestones=lr_decay, gamma=self.config.gamma)

            
            return [
                {"optimizer": self.optimizer_flow, "lr_scheduler": optimizer_flow_scheduler, "monitor": "val_loss"}
            ]
        else:
            return [self.optimizer_flow]
    """
    
    #def configure_optimizers(self):
    #    optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
    #    return optimizer
    
    def configure_optimizers(self):
        initial_lr = 0.01#self.config.lr
        optimizer = torch.optim.Adam(self.parameters(), lr=initial_lr)
        
        if self.config.use_lr_decay:
            lr_decay_steps = [int(epoch) for epoch in self.config.lr_decay.split(',')]
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_steps, gamma=self.config.gamma),
                'name': 'learning_rate',  # Optional: Give the scheduler a name for logging
                'interval': 'epoch',  # The unit of the scheduler's step size
                'frequency': 1,  # The frequency of the scheduler step
                'reduce_on_plateau': False,  # For ReduceLROnPlateau scheduler
                # 'monitor': 'val_loss'  # If your scheduler needs to look at a specific value, e.g., val_loss for ReduceLROnPlateau
            }
            return [optimizer], [lr_scheduler]
        
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
                      accelerator='gpu', devices=-1, profiler=profiler, log_every_n_steps=10)
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()