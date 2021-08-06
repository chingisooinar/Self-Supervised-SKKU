#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 18:54:17 2021

@author: nuvilabs
"""
import torch
import torch.nn as nn
import timm
import os
import math
from scipy.special import binom
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.models as models


def create_model(args):
    if args.model == 'googlenet':
        module = models.googlenet(pretrained=True)
        num_ftrs = module.fc.in_features
        module.fc = nn.Linear(num_ftrs, args.low_dim)
    elif args.model == 'vgg16':
        module = models.vgg16(pretrained=True)
        num_ftrs = module.classifier[-1].in_features
        module.classifier[-1] = nn.Linear(num_ftrs, args.low_dim)
    elif args.model == 'vgg19':
        module = models.vgg19(pretrained=True)
        num_ftrs = module.classifier[-1].in_features
        module.classifier[-1] = nn.Linear(num_ftrs, args.low_dim)
    elif args.model == 'vgg19_bn':
        module = models.vgg19_bn(pretrained=True)
        num_ftrs = module.classifier[-1].in_features
        module.classifier[-1] = nn.Linear(num_ftrs, args.low_dim)
    elif args.model == 'inception_v3':
        module = models.inception_v3(pretrained=True)
        num_ftrs = module.fc.in_features
        module.fc = nn.Linear(num_ftrs, args.low_dim)
    elif args.model == 'alexnet':
        module = models.alexnet(pretrained=True)
        num_ftrs = module.classifier[-1].in_features
        module.classifier[-1] = nn.Linear(num_ftrs, args.low_dim)
    elif args.model == 'resnet18':
        module = models.resnet18(pretrained=True)
        num_ftrs = module.fc.in_features
        module.fc = nn.Linear(num_ftrs, args.low_dim)
    elif args.model == 'resnext50':
        module = models.resnext50_32x4d(pretrained=True)
        num_ftrs = module.fc.in_features
        module.fc = nn.Linear(num_ftrs, args.low_dim)
    elif args.model == 'densenet161':
        module = models.densenet161(pretrained=True)
        num_ftrs = module.classifier.in_features
        module.classifier = nn.Linear(num_ftrs, args.low_dim)
    elif args.model == 'mobilenet_v2':
        module = models.mobilenet_v2(pretrained=True)
        num_ftrs = module.classifier[-1].in_features
        module.classifier[-1] = nn.Linear(num_ftrs, args.low_dim)
    elif args.model == 'resnet50':
        module = models.resnet50(pretrained=True)
        num_ftrs = module.fc.in_features
        module.fc = nn.Linear(num_ftrs, args.low_dim)
    return module


class EmbedderWrapper:

    def __init__(self, args, n_classes):
        print('==> Building model..')

        self.module = create_model(args)
        self.head = None
        self.training = False
        self.device = None
        self.dim = args.low_dim

        #print(self.module)

    def to_device(self, device):
        if device == 'cuda':
            self.module = torch.nn.DataParallel(self.module, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True
        self.module.to(device)
        self.device = device

    def get_parameters(self, args):
        return self.module.parameters()

    def resume(self, args, name):
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/'+name)
        self.module.load_state_dict(checkpoint['net'])

    def save_model(self, args):
        torch.save(self.module, args.instance+'_model.pth')
        torch.save(self.head.kernel, args.instance+'_kernel.pt')

    def train(self, mode=True):
        self.training = mode
        self.module.train()

    def eval(self, mode=False):
        self.training = mode
        self.module.eval()

    def __call__(self, x, targets=None, test=False):
        outputs = self.module(x)
        return outputs
