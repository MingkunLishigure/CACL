from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform1=None,transform2=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform1 is not None:
            img1 = self.transform1(img)
            img2 = self.transform2(img)
            

        return img1, img2, fname, pid, camid, index
