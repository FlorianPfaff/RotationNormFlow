from os.path import join
from dataset.lib.Dataset_Base import Dataset_Base
import torch
from pytorch3d import transforms as trans
from dataset.dataloader_utils import MixDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

cate10 = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
          'monitor', 'night_stand', 'sofa', 'table', 'toilet']

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

def get_dataloader_modelnet(collection, config, aug=None):
    datasets = []
    for category in cate10:
        dset = ModelNetDataset(
            config.data_dir, category, collection=collection, net_arch=config.net_arch, aug=aug)
        datasets.append(dset)

    entire_dataset = MixDataset(datasets)

    return entire_dataset, datasets, cate10


class PrecomputedFeaturesDataset(Dataset):
    def __init__(self, features_path, metadata_path):
        with np.load(features_path) as data:
            self.features = data['features']
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return a dictionary with all the metadata for the idx entry
        return {
            'feature': self.features[idx],
            **{key: self.metadata[key][idx] for key in self.metadata}
        }


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


class PrecomputedModelNetDataModule(LightningDataModule):
    def __init__(self, features_paths, metadata_paths, batch_size):
        super().__init__()
        # features_paths and metadata_paths are expected to be dicts with keys 'train', 'val', 'test'
        self.features_paths = features_paths
        self.metadata_paths = metadata_paths
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # Setup datasets for each stage
        if stage == 'fit' or stage is None:
            self.train_dataset = PrecomputedFeaturesDataset(self.features_paths['train'], self.metadata_paths['train'])
        if stage == 'validate' or stage is None:
            self.val_dataset = PrecomputedFeaturesDataset(self.features_paths['val'], self.metadata_paths['val'])
        if stage == 'test' or stage is None:
            self.test_dataset = PrecomputedFeaturesDataset(self.features_paths['test'], self.metadata_paths['test'])

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup('fit')
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup('validate')
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        if self.test_dataset is None:
            self.setup('test')
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
