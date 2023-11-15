from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from dataset.lib.Dataset_Base import Dataset_Base  # Ensure this import is correct
import torch
from pytorch3d import transforms as trans
import pickle
import numpy as np

class ModelNetDataset(Dataset_Base):
    def __init__(
        self, data_dir, category, collection="train", net_arch="alexnet", aug=None
    ):
        super(ModelNetDataset, self).__init__(
            data_dir, category, collection, net_arch, 1.0
        )
        self.aug = aug

    def __getitem__(self, idx):
        rc = self.recs[idx]
        cate = rc.category
        img_id = rc.img_id
        quat = rc.so3.quaternion
        quat = torch.from_numpy(quat)
        rot_mat = trans.quaternion_to_matrix(quat)

        img = self._get_image(rc)
        img = torch.from_numpy(img)

        if self.aug is not None:
            img = self.aug(img)

        sample = dict(
            idx=idx,
            cate=self.cate2ind[cate],
            quat=quat,
            rot_mat=rot_mat,
            img=img,
            img_id=img_id,
        )

        return sample

class ModelNetDataModule(LightningDataModule):
    def __init__(self, config, batch_size, num_workers, data_dir, net_arch="vgg16"):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.net_arch = net_arch
        self.cate10 = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_datasets = self._create_datasets("train")
            self.entire_train_dataset = ConcatDataset(self.train_datasets)

        if stage == "test" or stage is None:
            self.test_datasets = self._create_datasets("test")
            self.entire_test_dataset = ConcatDataset(self.test_datasets)

    def _create_datasets(self, collection):
        datasets = []
        for category in self.cate10:
            dataset = ModelNetDataset(self.data_dir, category, collection=collection, net_arch=self.net_arch, aug=None)
            datasets.append(dataset)
        return datasets

    def train_dataloader(self):
        return DataLoader(self.entire_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        # Return the test dataset DataLoader as validation DataLoader
        return DataLoader(self.entire_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.entire_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def individual_train_dataloaders(self):
        return [DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True) for dataset in self.train_datasets]

    def individual_test_dataloaders(self):
        return [DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True) for dataset in self.test_datasets]

class TrainFeaturesPrecomputedModelNetDataset(ModelNetDataset):
    def __init__(
        self, features_path, metadata_path, data_dir, category, collection="train", net_arch="alexnet", aug=None
    ):
        if collection == 'test':
            super().__init__(data_dir, category, collection, net_arch, aug)
        self.features_path = features_path
        self.metadata_path = metadata_path
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            with np.load(self.features_path['train']) as data:
                self.features = data['features']
            with open(self.metadata_path['train'], 'rb') as f:
                self.metadata = pickle.load(f)
        
        if stage == "test" or stage is None:
            raise NotImplementedError('')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return a dictionary with all the metadata for the idx entry
        return {
            'feature': self.features[idx],
            **{key: self.metadata[key][idx] for key in self.metadata}
        }
        
        
class TrainFeaturesPrecomputedModelNetDataModule(ModelNetDataModule):
    def __init__(self, features_path, metadata_path, config, batch_size, num_workers, data_dir, net_arch="vgg16"):
        super().__init__(config, batch_size, num_workers, data_dir, net_arch)
        self.features_path = features_path
        self.metadata_path = metadata_path
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.entire_train_dataset = TrainFeaturesPrecomputedModelNetDataset(self.features_path, self.metadata_path, self.batch_size,
                                                                         self.num_workers, self.data_dir, self.net_arch)
            self.entire_train_dataset.setup(stage='fit')
        if stage == "test" or stage is None:
            super().setup(stage='test')