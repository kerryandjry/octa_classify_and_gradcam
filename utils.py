import math
import random
import sys

import cv2
import numpy as np
from tqdm import tqdm
from torchvision import transforms

argument = transforms.Compose([transforms.RandomHorizontalFlip(p=0.7),
                              transforms.RandomVerticalFlip(p=0.7),
                              transforms.RandomRotation(degrees=(0, 180))])


def train_one_epoch(model, optimizer, data_loader, device, epoch) -> (float, float):
    model.train()
    loss_function = torch.nn.BCEWithLogitsLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):

        images, labels = data

        if random.random() > 0.7:
            images = argument(images)

        sample_num += images.shape[0]

        pred = model(images.to(device))
        class_pred = torch.tensor(torch.sigmoid(pred))
        class_pred[torch.where(class_pred >= 0.37)] = 1
        class_pred[torch.where(class_pred < 0.37)] = 0

        accu_num += torch.eq(class_pred, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.5f}".format(epoch, accu_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def val_one_epoch(model, data_loader, device) -> float:
    model.eval()
    val_num = torch.zeros(1).to(device)
    val_sum = 0

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        val_sum += images.shape[0]

        pred = model(images.to(device))
        class_pred = torch.tensor(torch.sigmoid(pred))
        class_pred[torch.where(class_pred >= 0.37)] = 1
        class_pred[torch.where(class_pred < 0.37)] = 0

        val_num += torch.eq(class_pred, labels.to(device)).sum()
    return val_num.item() / val_sum


def random_perspective(img, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2

    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2
    C[1, 2] = -img.shape[0] / 2

    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    M = T @ S @ R @ P @ C
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    return img


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss