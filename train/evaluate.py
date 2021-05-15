import sys
sys.path.insert(1, '../UnsupervisedLearning-JigsawPuzzle')

import torch
import torch.nn as nn

import torchvision
import os
import cv2
from torch.autograd import Variable
from dataset_factory.data_loader import AnimalDataset
from dataset_factory.data_utils import get_all_imgs
from models.AlexNet import AlexNet
from config import Config
import math
from tqdm import tqdm
from train.train_utils import prepare_dataloader
from torch.utils.data import DataLoader
from models.AlexNet import AlexNet

if __name__ == '__main__':
    data = get_all_imgs('../UnsupervisedLearning-JigsawPuzzle/dataset/test', return_label=True)
    print('Found unlabeled {} images'.format(len(data)))
    test_ds = AnimalDataset(data)
    test_loader = DataLoader(test_ds, batch_size = 16, num_workers = 4)
    net = AlexNet().cuda()
    net.load('../UnsupervisedLearning-JigsawPuzzle/save_model/Alexnet.pth')

    net.eval()
    total_acc = 0

    pbar = tqdm(enumerate(test_loader), total = len(test_loader))
    for step, (images, labels) in pbar:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        outputs = net(images)

        preds = torch.argmax(outputs, 1).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()

        acc = (preds == targets).mean()*100
        total_acc += acc

        description = f'Acc: {total_acc/(step+1):.6}'
        pbar.set_description(description)
    
    print('Accuracy on unlabeled set:', total_acc/(step+1))