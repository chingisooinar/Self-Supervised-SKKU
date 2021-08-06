#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:38:36 2021

@author: nuvilabs
"""
import numpy as np
import torch
import os

import pickle 


class DatasetLoader:
    def __init__(self, args, trainset, testset):
        self.directory = './data/' + args.instance
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)
        self.trainloader = trainset
        self.testloader = testset
