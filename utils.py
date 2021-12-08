import sys

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
