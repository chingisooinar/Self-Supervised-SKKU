import codecs
import os

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms


class SSL_Dataset(data.Dataset):
    def __init__(
        self,
        root,
        mode,
        transform=None,
        data=None,
        targets=None,
        **kwargs
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.mode = mode
        if isinstance(transform, tuple):
            self.weak_t, self.strong_t = transform
        if mode not in ['unlabeled_train', 'test']:
            self.data, self.targets = data, torch.LongTensor(targets)
        else:
            self.data = data
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.ToTensor = transforms.ToTensor()

        self.t = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.Lambda(lambda x: np.array(x,dtype=np.uint8)),
        ])
        #################### EDIT HERE ####################
        ###################################################
   
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode in ['labeled_train']: 
            impath, target = self.data[index], self.targets[index]
            img = Image.open('../kaggle_data'+ impath).convert('RGB')
            img = self.t(img)
           # print(type(img))
            if self.transform is not None:
                sample = self.transform(image = img)
                img = sample['image']
            target = target.long()
            img = self.ToTensor(img)
            img = self.normalizer(img)
            return img, target, index
        elif self.mode in ['unlabeled_train']:
            impath = self.data[index]
            img = Image.open('../kaggle_data'+ impath).convert('RGB')
            if self.transform is not None:
                weak_x = self.weak_t(img)
                strong_x = self.strong_t(img)

            weak_x = self.ToTensor(weak_x)
            weak_x = self.normalizer(weak_x)
            strong_x = self.ToTensor(strong_x)
            strong_x = self.normalizer(strong_x)
            return weak_x, strong_x
        elif self.mode in ['test']: 
            impath = self.data[index]
            #print(impath)
            img = Image.open('../kaggle_data'+ impath).convert('RGB')
            img = self.t(img)
            img = self.ToTensor(img)
            img = self.normalizer(img)
            return img
        #################### EDIT HERE ####################
        ###################################################



