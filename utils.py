#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:07:08 2021

@author: nuvilabs
"""
import os
import albumentations as A
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import pandas as pd
from model import EmbedderWrapper

def get_model(args):
    model = EmbedderWrapper(args, 10)
    device = 'cuda'
    model.to_device(device)
    return model


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.shape[1]
        w = img.shape[2]

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        #mask = torch.from_numpy(mask)
        #mask = mask.expand_as(img)
        
        img = img * mask

        return img


def save_prediction(weight_path, pred, test_idx):
    sub_id = pd.read_csv('../kaggle_data/annotation/test_id.csv', delimiter=',') #idx 기준

    sub_df = pd.DataFrame()
    sub_df['id'] = test_idx
    sub_df['label'] = pred
    sub_df = sub_df.set_index('id', drop=False)
    sub_df = sub_df.reindex(sub_id['id'])
    sub_df = sub_df.reset_index(drop=True)
    sub_df.to_csv(weight_path+'submission.csv', index=None)

    print('\nSubmission File Saved...!!')


def load_data(root, mode):
    data = []
    targets = []
    idx = []

    labeled_train = "kaggle_data/annotation/train_labeled_filelist.txt" 
    unlabeled_test = "kaggle_data/annotation/test_filelist.txt"
    unlabeled_train = "kaggle_data/annotation/train_unlabeled_filelist.txt" 
    if mode == 'labeled_train':
        flist = root + labeled_train
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                imgdata, clean_label = line.strip().split()
                data.append(imgdata)
                targets.append(float(clean_label))
            return data, targets
    elif mode == 'unlabeled_train':
        flist = root + unlabeled_train
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                imgdata = line.strip()
                data.append(imgdata)
            return data
    elif mode == 'test':
        flist = root + unlabeled_test
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                imgdata = line.strip()
                data.append(imgdata)
                idx.append(imgdata.split('/')[2][:-4])
            return idx, data


