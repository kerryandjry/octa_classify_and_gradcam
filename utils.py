import math
import random
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm


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
        if random.random() > 0.5:
            images = random_perspective(images)
        sample_num += images.shape[0]

        pred = model(images.to(device))
        class_pred = (pred > 0.5) * 1.
        int_labels = (np.array(labels))
        print(class_pred, int_labels, class_pred == int_labels)
        # accu_num += torch.eq(pred_classes, labels.to(device)).sum()

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
    val_num = torch.zeros(1).to(device)
    val_sum = 0

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        val_sum += images.shape[0]
        pred = model(images.to(device))
        class_pred = torch.tensor(pred > 0.5 * 1.)
        val_num += torch.eq(class_pred, labels.to(device)).sum()

    return val_num / val_sum


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