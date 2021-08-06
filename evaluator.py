#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:58:15 2021

@author: nuvilabs
"""
import torch


class Evaluator:
    def __init__(self, args, net, dataloader, ndata=None):
        self.epochs = args.epochs
        self.net = net
        self.lr = args.lr
        self.ndata = ndata
        self.testloader = dataloader
        self.device = net.device if not isinstance(net, list) else net[0].device
        self.args = args
        if args.test_only or len(args.resume) > 0:
            if isinstance(net, list):
                ckpts = args.resume.split(',')
                for i in range(len(net)):
                    self.net[i].resume(args, ckpts[i])
            else:
                self.net.resume(args, args.resume)

    def evaluate_ensemble(self):
        for net in self.net:
            net.eval()
        test_preds = []
        with torch.no_grad():
            for batch_data in self.testloader: 
                batch_x = batch_data 
                inputs = batch_x.to(self.device)
                probs = torch.zeros((inputs.shape[0], 10)).cuda()
                for net in self.net:
                    outputs = net(inputs)
                    probs += torch.softmax(outputs, dim=-1)
                #probs /= len(self.net)
                _, predicted = torch.max(probs, 1)
                if self.device == 'cuda':
                    test_preds += predicted.detach().cpu().numpy().tolist()
                else:
                    test_preds += predicted.detach().numpy().tolist()
        return test_preds

    def evaluate(self):
        self.net.eval()
        test_preds = []
        with torch.no_grad():
            for batch_data in self.testloader: 
                batch_x = batch_data 
                inputs = batch_x.to(self.device)

                outputs = self.net(inputs)
                _, predicted = torch.max(outputs, 1)
                if self.device == 'cuda':
                    test_preds += predicted.detach().cpu().numpy().tolist()
                else:
                    test_preds += predicted.detach().numpy().tolist()
        return test_preds
