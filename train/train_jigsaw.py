import sys
sys.path.insert(1, '../UnsupervisedLearning-JigsawPuzzle')

import torch
import torch.nn as nn
import torchvision
import os
import cv2
from torch.autograd import Variable
from dataset_factory.data_loader import JigsawDataset
from dataset_factory import data_utils
from models.AlexNet import JigsawAlexNet
from config import Config
import math
from metric import accuracy as acc_metric
from tqdm import tqdm
from train.train_utils import prepare_dataloader
import argparse

parser = argparse.ArgumentParser(description='Train JigsawPuzzle Classifer')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--gamma', default=0.3, type=float, help='gamma for StepLR')
parser.add_argument('--period', default=30, type=int, help='period range for StepLR')
parser.add_argument('--pretrained', default=None, type=str, help='Path to pretrained model')
parser.add_argument('--checkpoint', default=None, type=str, help='Path to save checkpoint model')
parser.add_argument('--train_csv', default='../UnsupervisedLearning-JigsawPuzzle/dataset/csv/train.csv', type=str, help='Path to train.csv')
parser.add_argument('--valid_csv', default='../UnsupervisedLearning-JigsawPuzzle/dataset/csv/valid.csv', type=str, help='Path to valid.csv')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs for training')
parser.add_argument('--train_batch', default=16, type=int, help='train batch size')
parser.add_argument('--valid_batch', default=16, type=int, help='valid batch size')
parser.add_argument('--init_acc', default=0, type=float, help='initial accuracy for training')
parser.add_argument('--result', default=None, type=str, help='Path to save result log')

args = parser.parse_args()

def train_one_epoch(epoch, net, train_loader, loss_fc, optimizer):
    net.train()
    total_loss = 0
    total_acc = 0
    pbar = tqdm(enumerate(train_loader), total = len(train_loader))
    for step, (images, labels, orginal) in pbar:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(images)

        preds = torch.argmax(outputs, 1).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()

        acc = (preds == targets).mean()*100

        loss = loss_fc(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc

        description = f'epoch {epoch} || Loss: {total_loss/(step+1):.6f} | Acc: {total_acc/(step+1):.6}'
        pbar.set_description(description)

def valid_one_epoch(epoch, net, valid_loader, loss_fc):
    net.eval()
    total_loss = 0
    total_acc = 0

    pbar = tqdm(enumerate(valid_loader), total = len(valid_loader))
    for step, (images, labels, orginal) in pbar:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        outputs = net(images)

        preds = torch.argmax(outputs, 1).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()

        acc = (preds == targets).mean()*100

        loss = loss_fc(outputs, labels)

        total_loss += loss.item()
        total_acc += acc

        description = f'epoch {epoch} || Loss: {total_loss/(step+1):.6f} | Acc: {total_acc/(step+1):.6}'
        pbar.set_description(description)
    
    return total_acc/(step+1)


if __name__ == '__main__':
    train_loader, valid_loader = prepare_dataloader(JigsawDataset, args.train_csv, args.valid_csv, args.train_batch, args.valid_batch)

    net = JigsawAlexNet().cuda()
    # net.apply(weights_init)

    if args.pretrained:
        net.load_state_dict(torch.load(args.pretrained))
        print('Load pretrained model successfully!')
    
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.period, args.gamma, verbose = True)

    if args.result:
        f = open(args.result, "w")
    # training
    print('='*30)
    print('Start training ...')
    best_acc = args.init_acc
    for epoch in range(args.epochs):
        train_one_epoch(epoch, net, train_loader, loss, optimizer)
        with torch.no_grad():
            acc = valid_one_epoch(epoch, net, valid_loader, loss)
            if acc > best_acc:
                best_acc = acc
                torch.save(net.state_dict(), args.checkpoint)
                print('Save checkpoint ... Best accuracy {:.3f}'.format(best_acc))
                if args.result:
                    f.write("Epoch: " + str(epoch) + ', best acc save: ' + str(best_acc) + '\n')
        scheduler.step()
    if args.result:
        f.close()
