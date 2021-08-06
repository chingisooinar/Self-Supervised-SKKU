#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 22:16:10 2021

@author: nuvilabs
"""
import torch.nn.functional as F
import torch


class FixMatchCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(FixMatchCrossEntropy, self).__init__()

    def forward(self, outputs, targets, mask=None):
        if mask is not None:
            return (F.cross_entropy(outputs, targets, reduction='none') * mask).mean()
        else:
            return F.cross_entropy(outputs, targets, reduction='mean')