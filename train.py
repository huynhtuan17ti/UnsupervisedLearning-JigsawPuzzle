import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import os
import cv2
from torch.autograd import Variable
from dataset_factory.data_loader import get_all_imgs, JigsawDataset
from models.AlexNet import JigsawAlexNet
from config import Config
import math
from metric import accuracy as acc_metric
from tqdm import tqdm

cfg = Config()

def prepare_dataloader():
    labeled_img = get_all_imgs(cfg.train_path)
    print('Found {} images in labeled set'.format(len(labeled_img)))

    train_sz = math.ceil(len(labeled_img)*(1 - cfg.valid_ratio))
    train_img, valid_img = labeled_img[:train_sz], labeled_img[train_sz:]

    print('Train set: {} images'.format(len(train_img)))
    print('Valid set: {} images'.format(len(valid_img)))

    train_ds = JigsawDataset(train_img)
    valid_ds = JigsawDataset(valid_img)

    train_loader = DataLoader(train_ds, batch_size=cfg.train_batch, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.valid_batch)

    return train_loader, valid_loader

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

        prec1 = acc_metric.compute_accuracy(outputs.cpu().data, labels.cpu().data, topk=(1, 5))

        acc = prec1[0]

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
    pbar = tqdm(enumerate(train_loader), total = len(train_loader))
    for step, (images, labels, orginal) in pbar:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(images)

        prec1 = acc_metric.compute_accuracy(outputs.cpu().data, labels.cpu().data, topk=(1, 5))

        acc = prec1[0]

        loss = loss_fc(outputs, labels)

        total_loss += loss.item()
        total_acc += acc

        description = f'epoch {epoch} || Loss: {total_loss/(step+1):.6f} | Acc: {total_acc/(step+1):.6}'
        pbar.set_description(description)
    
    return total_acc/(step+1)


if __name__ == '__main__':
    train_loader, valid_loader = prepare_dataloader()

    net = JigsawAlexNet().cuda()
    if cfg.pretrain:
        None
        # do something
    
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = cfg.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1, verbose = True)

    # training
    print('='*30)
    print('Start training ...')
    best_acc = 0
    for epoch in range(cfg.num_epochs):
        train_one_epoch(epoch, net, train_loader, loss, optimizer)
        with torch.no_grad():
            acc = valid_one_epoch(epoch, net, valid_loader, loss)
            if acc > best_acc:
                best_acc = acc
                net.save(cfg.checkpoint_path)
                print('Save checkpoint ... Best accuracy {:.3f}'.format(best_acc))
        scheduler.step()
