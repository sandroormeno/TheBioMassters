import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .d_s import Dataset
import albumentations as A

class DataModule(pl.LightningDataModule):
    def __init__(self, s1=True, s2=True, s3=True, month = None, batch_size=32, num_workers=0, pin_memory=False, val_size=0, random_state=42):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_size = val_size
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.month = month
        self.random_state = random_state

        self.transform = A.Compose([
                    #A.Transpose(),
                    A.Rotate(),
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                    #A.RandomRotate90(),
                    A.Flip(),
                ],additional_targets = {'image0': 'image', 'image1': 'image'}) #{'image0': 'image', 'image1': 'image'}

    def setup(self, stage=None):
        # read json files
        train = pd.read_json('https://raw.githubusercontent.com/sandroormeno/TheBioMassters/main/train.json')
        test = pd.read_json('https://raw.githubusercontent.com/sandroormeno/TheBioMassters/main/test.json')
        
        val = None
        # validation split
        if self.val_size > 0:
            train, val = train_test_split(train, test_size=self.val_size, random_state=self.random_state)
        # generate datastes
        #self.ds_train = Dataset(train.filename.values,self.s1, self.s2, self.s3, train.label.values) # sin transform
        self.ds_train = Dataset(train.filename.values,self.s1, self.s2, self.s3, self.month, train.label.values, self.transform)
        #self.ds_train = Dataset(train.filename.values,self.s1, self.s2, train.label.values) # for overfitting
        self.ds_val = None
        if val is not None:
            self.ds_val = Dataset(val.filename.values, self.s1, self.s2, self.s3, self.month, val.label.values)
        self.ds_test = Dataset(test.filename.values,self.s1, self.s2, self.s3, self.month, chip_ids = test.index.values)
        print('train:', len(self.ds_train))
        if self.ds_val is not None:
            print('val:', len(self.ds_val))
        print('test:', len(self.ds_test))
    def get_dataloader(self, ds, batch_size=None, shuffle=None):
        return DataLoader(
            ds,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            shuffle=shuffle if shuffle is not None else True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        ) if ds is not None else None

    def train_dataloader(self, batch_size=None, shuffle=True):
        return self.get_dataloader(self.ds_train, batch_size, shuffle)

    def val_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_val, batch_size, shuffle)

    def test_dataloader(self, batch_size=None, shuffle=False):
        return self.get_dataloader(self.ds_test, batch_size, shuffle)

def collate_fn(batch):

    s1s, s2s, s3s, labels = zip(*batch)
    s1s = torch.from_numpy(np.stack(s1s)) if s1s[0] is not None else None
    s2s = torch.from_numpy(np.stack(s2s)) if s2s[0] is not None else None
    s3s = torch.from_numpy(np.stack(s3s)) if s3s[0] is not None else None
    return (s1s, s2s, s3s), labels if isinstance(labels[0], str) else torch.from_numpy(np.stack(labels))

