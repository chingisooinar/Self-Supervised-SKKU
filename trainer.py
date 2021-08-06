#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 18:44:07 2021

@author: nuvilabs
"""
import torch
from lib.focal_loss import FocalLoss
import torch.nn as nn
import torch.optim as optim
from lib.utils import AverageMeter
import time
import sys
import os
import wandb
from test import evaluate
from losses import FixMatchCrossEntropy
train_labels_ = None


def create_optimizer(args, net):
    parameters = net.get_parameters(args)
    if args.fixmatch and False:
        optimizer = optim.SGD(parameters, lr=args.lr, weight_decay=5e-4, momentum=0.9)
    else:
        optimizer = optim.Adam(parameters, lr=args.lr)

    return optimizer


def create_criterion(args, ndata=None):

    if args.focal:
        print('Focal Loss')
        criterion = FocalLoss(gamma=3)
    elif args.uda:
        return nn.KLDivLoss(reduction='batchmean'), nn.CrossEntropyLoss()
    else:
        print('CrossEntropyLoss')
        criterion = nn.CrossEntropyLoss()
    return criterion


def get_train_labels(trainloader, device='cuda'):
    global train_labels_
    if train_labels_ is None:
        print("=> loading all train labels")
        train_labels = []
        for i, (_, label, index) in enumerate(trainloader):
            train_labels.extend(label.cpu().numpy().tolist())
            if i % 10000 == 0:
                print("{}/{}".format(i, len(trainloader)))
        train_labels_ = torch.tensor(train_labels).long().to(device)
        torch.save(train_labels_, 'labels.pt')
    print(train_labels_)
    return train_labels_


class Trainer:
    def __init__(self, args, net, dataloader):
        self.epochs = args.epochs
        self.optimizer = create_optimizer(args, net)
        self.net = net
        self.lr = args.lr
        self.trainloader = dataloader.trainloader
        self.testloader = dataloader.testloader
        self.device = net.device
        self.wandb = wandb.init(config=args, project="metric-universal", name=args.instance)
        self.args = args
        self.directory = './data/' + args.instance


        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)
        self.criterion = create_criterion(args)
        if net.device:
            self.criterion.to(net.device)
        self.best_acc = 0.
        self.start_epoch = 1
        if args.test_only or len(args.resume) > 0:
            self.resume(args)
            self.net.resume(args, args.resume)

    def resume(self, args):
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/'+args.resume)
        self.best_acc = checkpoint['acc']
        self.start_epoch = checkpoint['epoch']

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr
        if epoch >= 80:
            lr = self.lr * (0.1 ** ((epoch-80) // 40))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def report(self, epoch, batch_idx, batch_time, data_time, loss, loader):
        if batch_idx % 10 == 0:
            print('\n'+'Epoch: [{}][{}/{}]'
              'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
              'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
              'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
              epoch, batch_idx, len(loader), batch_time=batch_time, data_time=data_time, loss=loss))
        else:
            print('\r'+'Epoch: [{}][{}/{}]'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
              epoch, batch_idx, len(loader), batch_time=batch_time, data_time=data_time, loss=loss),end="")
            sys.stdout.flush()

    def train_epoch(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.adjust_learning_rate(self.optimizer, epoch)
        train_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        # switch to train mode
        self.net.train()

        end = time.time()
        for batch_idx, (inputs, targets, indexes) in enumerate(self.trainloader):
            data_time.update(time.time() - end)
            inputs, targets, indexes = inputs.to(self.device), targets.to(self.device), indexes.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.report(epoch, batch_idx, batch_time, data_time, train_loss, self.trainloader)
        return train_loss.avg

    def val_epoch(self, epoch):
        print('\nEpoch: %d' % epoch)
        test_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        # switch to train mode
        self.net.eval()

        end = time.time()
        with torch.set_grad_enabled(False):
            for batch_idx, (inputs, targets, indexes) in enumerate(self.testloader):
                data_time.update(time.time() - end)
                inputs, targets, indexes = inputs.to(self.device), targets.to(self.device), indexes.to(self.device)

                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss.update(loss.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                self.report(epoch, batch_idx, batch_time, data_time, test_loss, self.testloader)
        return test_loss.avg

    def train_cycle(self):
        best_acc = self.best_acc
        acc = 0.
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            train_loss = self.train_epoch(epoch)
            acc = evaluate(epoch=epoch, net=self.net, testloader=self.testloader)

            report = {'epoch': epoch, 'f1': acc, 'train_loss': train_loss}

            test_loss = self.val_epoch(epoch)
            report['valid_loss'] = test_loss
            self.wandb.log(report)
            if acc >= best_acc or epoch % 10 == 0:
                self.save(acc, epoch, best_acc)
                best_acc = acc if acc > best_acc else best_acc
            print('best accuracy: {:.2f}'.format(best_acc*100))

    def train(self):
        self.wandb.watch(self.net.module)
        self.train_cycle()
        self.module.resume(self.args, self.best_checkpoint)

    def save(self, acc, epoch, best_acc):
        print('Saving..')
        state = {
            'net': self.net.module.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        is_best = 'best_' if acc > best_acc else ''
        title = 'ckpt_' + is_best + self.args.instance + '_' + str(self.args.low_dim) \
            + '_' + str(acc) + '_' + str(epoch) + '.t7'
        if acc > best_acc:
            self.best_checkpoint = title
        torch.save(state, './checkpoint/' + title)


class FixMatchTrainer(Trainer):
    def __init__(self, args, net, dataloader, unlabeledloader):
        self.epochs = args.epochs
        self.optimizer = create_optimizer(args, net)
        self.net = net
        self.lr = args.lr
        self.trainloader = dataloader.trainloader
        self.testloader = dataloader.testloader
        self.unlabeledloader = unlabeledloader
        self.unlabeled_iter = iter(self.unlabeledloader)
        self.device = net.device
        self.wandb = wandb.init(config=args, project="metric-universal", name=args.instance)
        self.args = args
        self.directory = './data/' + args.instance
        self.threshold = args.threshold
        self.lambda_u = args.lambda_u
        self.temp = args.T
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        self.criterion = FixMatchCrossEntropy()
        if net.device:
            self.criterion.to(net.device)
        self.best_acc = 0.
        self.start_epoch = 1
        if args.test_only or len(args.resume) > 0:
            self.net.resume(args, args.resume)

    def train_epoch(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.adjust_learning_rate(self.optimizer, epoch)
        train_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        # switch to train mode
        self.net.train()

        end = time.time()
        for batch_idx, (inputs, targets, indexes) in enumerate(self.trainloader):
            data_time.update(time.time() - end)
            try:
                uinputs_weak, uinputs_strong = next(self.unlabeled_iter)
            except:
                self.unlabeled_iter = iter(self.unlabeledloader)
                uinputs_weak, uinputs_strong = next(self.unlabeled_iter)
            uinputs_weak, uinputs_strong = uinputs_weak.to(self.device), uinputs_strong.to(self.device)

            inputs, targets, indexes = inputs.to(self.device), targets.to(self.device), indexes.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            pseudo = self.net(uinputs_weak)
            pseudo = torch.softmax(pseudo.detach() / self.temp, dim=-1)
            max_probs, targets_u = torch.max(pseudo, dim=-1)
            mask = max_probs.ge(self.threshold).float()
            if (mask * 1.).sum() == 0:
                print('No pseudo labels were produced!')
            logits_u_s = self.net(uinputs_strong)

            unlabeled_loss = self.criterion(logits_u_s, targets_u, mask=mask)
            labeled_loss = self.criterion(outputs, targets)

            loss = labeled_loss + self.lambda_u * unlabeled_loss
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.report(epoch, batch_idx, batch_time, data_time, train_loss, self.trainloader)
        return train_loss.avg

    def val_epoch(self, epoch):
        print('\nEpoch: %d' % epoch)
        test_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        # switch to train mode
        self.net.eval()

        end = time.time()
        with torch.set_grad_enabled(False):
            for batch_idx, (inputs, targets, indexes) in enumerate(self.testloader):
                data_time.update(time.time() - end)
                inputs, targets, indexes = inputs.to(self.device), targets.to(self.device), indexes.to(self.device)

                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss.update(loss.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                self.report(epoch, batch_idx, batch_time, data_time, test_loss, self.testloader)
        return test_loss.avg

    def train_cycle(self):
        best_acc = self.best_acc
        acc = 0.
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            train_loss = self.train_epoch(epoch)
            acc = evaluate(epoch=epoch, net=self.net, testloader=self.testloader)

            report = {'epoch': epoch, 'f1': acc, 'train_loss': train_loss}

            test_loss = self.val_epoch(epoch)
            report['valid_loss'] = test_loss
            self.wandb.log(report)
            if acc >= best_acc or epoch % 10 == 0:
                self.save(acc, epoch, best_acc)
                best_acc = acc if acc > best_acc else best_acc
            print('best accuracy: {:.2f}'.format(best_acc*100))


class UDATrainer(Trainer):
    def __init__(self, args, net, dataloader, unlabeledloader):
        self.epochs = args.epochs
        self.optimizer = create_optimizer(args, net)
        self.net = net
        self.lr = args.lr
        self.trainloader = dataloader.trainloader
        self.testloader = dataloader.testloader
        self.unlabeledloader = unlabeledloader
        self.unlabeled_iter = iter(self.unlabeledloader)
        self.device = net.device
        self.wandb = wandb.init(config=args, project="metric-universal", name=args.instance)
        self.args = args
        self.directory = './data/' + args.instance
        self.threshold = args.threshold
        self.lambda_u = args.lambda_u
        self.temp = args.T
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        self.uda_criterion, self.cross_entropy = create_criterion(args)
        if net.device:
            self.uda_criterion.to(net.device)
            self.cross_entropy.to(net.device)
        self.best_acc = 0.
        self.start_epoch = 1
        if args.test_only or len(args.resume) > 0:
            self.net.resume(args, args.resume)

    def train_epoch(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.adjust_learning_rate(self.optimizer, epoch)
        train_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        # switch to train mode
        self.net.train()

        end = time.time()
        for batch_idx, (inputs, targets, indexes) in enumerate(self.trainloader):
            data_time.update(time.time() - end)
            try:
                uinputs_weak, uinputs_strong = next(self.unlabeled_iter)
            except:
                self.unlabeled_iter = iter(self.unlabeledloader)
                uinputs_weak, uinputs_strong = next(self.unlabeled_iter)
            uinputs_weak, uinputs_strong = uinputs_weak.to(self.device), uinputs_strong.to(self.device)

            inputs, targets, indexes = inputs.to(self.device), targets.to(self.device), indexes.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            pseudo = self.net(uinputs_weak)
            pseudo_probs = torch.softmax(pseudo.detach(), dim=-1)


            logits_u_s = self.net(uinputs_strong)
            strong_u_probs = torch.log_softmax(logits_u_s, dim=-1)
            unlabeled_loss = self.uda_criterion(strong_u_probs, pseudo_probs)
            labeled_loss = self.cross_entropy(outputs, targets)

            loss = labeled_loss + self.lambda_u * unlabeled_loss
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.report(epoch, batch_idx, batch_time, data_time, train_loss, self.trainloader)
        return train_loss.avg

    def val_epoch(self, epoch):
        print('\nEpoch: %d' % epoch)
        test_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        # switch to train mode
        self.net.eval()

        end = time.time()
        with torch.set_grad_enabled(False):
            for batch_idx, (inputs, targets, indexes) in enumerate(self.testloader):
                data_time.update(time.time() - end)
                inputs, targets, indexes = inputs.to(self.device), targets.to(self.device), indexes.to(self.device)

                outputs = self.net(inputs)
                loss = self.cross_entropy(outputs, targets)

                test_loss.update(loss.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                self.report(epoch, batch_idx, batch_time, data_time, test_loss, self.testloader)
        return test_loss.avg

    def train_cycle(self):
        best_acc = self.best_acc
        acc = 0.
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            train_loss = self.train_epoch(epoch)
            acc = evaluate(epoch=epoch, net=self.net, testloader=self.testloader)

            report = {'epoch': epoch, 'f1': acc, 'train_loss': train_loss}

            test_loss = self.val_epoch(epoch)
            report['valid_loss'] = test_loss
            self.wandb.log(report)
            if acc >= best_acc or epoch % 10 == 0:
                self.save(acc, epoch, best_acc)
                best_acc = acc if acc > best_acc else best_acc
            print('best accuracy: {:.2f}'.format(best_acc*100))

