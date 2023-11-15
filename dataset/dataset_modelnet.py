from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from dataset.lib.Dataset_Base import Dataset_Base  # Ensure this import is correct
import torch
from pytorch3d import transforms as trans

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
