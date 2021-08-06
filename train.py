#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:06:47 2021

@author: nuvilabs
"""
from __future__ import print_function
import copy
import torch
import argparse
import albumentations as A
from torch.utils.data import DataLoader
from dataloader import DatasetLoader
from trainer import Trainer, FixMatchTrainer, UDATrainer
import pickle
from evaluator import Evaluator
from utils import load_data, get_model, save_prediction
from dataset import SSL_Dataset
from RandAugment import RandAugment
from torchvision.transforms import transforms
import random
import numpy as np
LAST_ACC = 0
LAST_EPOCH = 0
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--m_2', default=0.5, type=float,
                    help='temperature parameter for softmax')
parser.add_argument('--m_1', default=1, type=int,
                    help='temperature parameter for softmax')
parser.add_argument('--b', default=32, type=int,
                    metavar='B', help='momentum for non-parametric updates')
parser.add_argument('--topk', default=200, type=int,
                    metavar='O', help='momentum for non-parametric updates')

parser.add_argument('--arcface', default=False, type=bool,
                    metavar='O', help='Class balancer')
parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]", default='ir_se', type=str)

parser.add_argument('--drop', default=False, type=bool, help='drop outliers')
parser.add_argument('--subcenters', default=1, type=int, help='# of subcenters')
parser.add_argument('--T', default=1, type=float, help='temperature')
parser.add_argument('--threshold', default=0.95, type=float, help='threshold')
parser.add_argument('--lambda_u', default=1., type=float, help='lambda_u')
parser.add_argument('--epochs', default=100, type=int, help='# of epochs')
parser.add_argument('--head_predict', default=False, type=bool, help='run test via kernel')
parser.add_argument('--model', default=None, type=str, help='path to valset npy array')
parser.add_argument('--save_features', default=False, type=bool, help='save features')
parser.add_argument('--instance', default=None, type=str, help='wandb project name')
parser.add_argument('--save_model', default=False, type=bool, help='save model')
parser.add_argument('--focal', default=False, type=bool, help='focal loss')
parser.add_argument('--fixmatch', default=False, type=bool, help='fixmatch')
parser.add_argument('--uda', default=False, type=bool, help='uda')
parser.add_argument('--ensemble', default=False, type=bool, help='uda')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.ImageCompression(always_apply=False, p=.45, quality_lower=69, quality_upper=100, compression_type=1),
        A.OneOf([
            A.ChannelShuffle(always_apply=False, p=0.5),
            A.ChannelDropout(always_apply=False, p=0.5, channel_drop_range=(1, 1), fill_value=0),
            A.RGBShift(always_apply=False, p=0.5, r_shift_limit=(-20, 20), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20)),
            ], p=0.3),
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.4),
            A.ElasticTransform(always_apply=False, p=0.3, alpha=4.5, sigma=67.11000061035156,
                               alpha_affine=38.2599983215332, interpolation=0, border_mode=1, value=(0, 0, 0),
                               mask_value=None, approximate=False),
        ], p=0.4),

        A.RandomGamma(always_apply=False, p=0.3, gamma_limit=(74, 153), eps=1e-07),
        A.GridDistortion(always_apply=False, p=0.3, num_steps=5,
                         distort_limit=(-0.30000001192092896, 0.30000001192092896), interpolation=1, border_mode=0,
                         value=(0, 0, 0), mask_value=None),
        A.CoarseDropout(always_apply=False, p=0.5, max_holes=8, max_height=8, max_width=8, min_holes=8,
                        min_height=8, min_width=8)
    ])

weak_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.Lambda(lambda x: np.array(x,dtype=np.uint8))
])
strong_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    RandAugment(2, 10),
    transforms.Lambda(lambda x: np.array(x,dtype=np.uint8)),
])
valid_ratio = 0.1

train_data_mode = 'labeled_train'
data, targets = load_data('../', train_data_mode)  # SSL_Dataset(root='../', transform=transform_train, mode=train_data_mode)
num_train = len(data)
num_valid = int(num_train * valid_ratio)
val_data = []
val_targets = []
counter = {}
indxs = []
f = open('val_data_googlenet.pkl', 'rb')
indxs = pickle.load(f)
for _ in range(num_valid):
    idx = indxs.pop(0)
    val_data.append(data.pop(idx))
    target = int(targets.pop(idx))
    counter[target] = counter.get(target, 0) + 1
    val_targets.append(target)
print(counter)
counter = {}
for target in targets:
    counter[int(target)] = counter.get(int(target), 0) + 1
print(counter)

if args.fixmatch or args.uda:
    udata = load_data('../', 'unlabeled_train')
    train_unlabeled_dataset = SSL_Dataset(root='../', transform=(weak_transform, strong_transform), mode='unlabeled_train', data=udata)
    unlabeled_trainloader = DataLoader(train_unlabeled_dataset, batch_size=args.b, shuffle=True)
    
    
#test_labeled_dataset = SSL_Dataset(root='../', transform=None, mode="test")
train_labeled_dataset = SSL_Dataset(root='../', transform=transform_train, mode=train_data_mode, data=data, targets=targets)
valid_labeled_dataset = SSL_Dataset(root='../', transform=None, mode=train_data_mode, data=val_data, targets=val_targets)

labeled_trainloader = DataLoader(train_labeled_dataset, batch_size=args.b, shuffle=True)
labeled_validloader = DataLoader(valid_labeled_dataset, batch_size=args.b, shuffle=False)
#labeled_testloader = DataLoader(test_labeled_dataset, batch_size=args.b, shuffle=False)
dataloader = DatasetLoader(args, labeled_trainloader, labeled_validloader)
if args.ensemble:
    assert ',' in args.model
    models = args.model.split(',')
    nets = []
    for name in models:
        args.model = name
        model = get_model(args)
        nets.append(model)
else:
    model = get_model(args)
if args.test_only:
    assert len(args.resume) > 0
    idx, testdata = load_data('../', 'test')
    test_dataset = SSL_Dataset(root='../', transform=None, mode="test", data=testdata)
    dataloader = DataLoader(test_dataset, batch_size=args.b, shuffle=False)
    evaluator = Evaluator(args, model if not args.ensemble else nets, dataloader)
    preds = evaluator.evaluate() if not args.ensemble else evaluator.evaluate_ensemble()
    save_prediction('./', preds, idx)
elif args.fixmatch or args.uda:
    #assert len(args.resume) > 0
    print('Semi-supervised...')
    trainer = FixMatchTrainer(args, model, dataloader, unlabeled_trainloader) if args.fixmatch else UDATrainer(args, model, dataloader,
                                                                                                               unlabeled_trainloader)
    trainer.train()
else:
    trainer = Trainer(args, model, dataloader)
    trainer.train()

