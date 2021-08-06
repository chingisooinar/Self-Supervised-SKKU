#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:51:25 2021

@author: nuvilabs
"""

import torch
import time
from lib.utils import AverageMeter
from sklearn.metrics import f1_score
import json
import math



def per_class(acc, labels, title):
    accs = {}
    for a, lbl in zip(acc, labels):
        accs[lbl] = a
    with open(title+'.json', 'w') as outfile:
        json.dump(accs, outfile)


def evaluate(epoch, net, testloader, train_labels=None, test=False):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()
    C = 10
    print(C)
    top1 = 0.
    f1 = 0.
    confusion_matrix = torch.zeros(C, C)

    with torch.no_grad():

        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda(non_blocking=True)
            bs = inputs.size(0)
            out = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            probs_sorted, predictions = torch.max(out, dim=1, keepdims=True)

            f1 += f1_score(predictions[:, 0].cpu().numpy(), targets.cpu().numpy(), average='micro')
            correct = predictions.eq(targets.data.view(-1, 1))
            cls_time.update(time.time() - end)
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()

            total += targets.size(0)
            for t, p in zip(targets.view(-1), predictions[:, 0].view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            if batch_idx % 1 == 0:
                print(f'Test [{total}/{testsize}]\t'
                      f'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                      f'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                      f'Top1: {top1 * 100. / total:.2f}\t'
                      f'F1 score: {f1/(batch_idx+1):.2f}')
    if test:
        per_class((confusion_matrix.diag()/confusion_matrix.sum(1)).cpu().numpy().tolist(),
                  testloader.dataset.classes, f'accuracy_{theta}')

    return f1/(batch_idx+1)