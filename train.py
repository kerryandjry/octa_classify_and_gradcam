import argparse
import os
import torch

from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import model
import swim_transformer_model
from dataset import MyData
from utils import train_one_epoch, val_one_epoch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
transform = transforms.Compose(
    [transforms.Resize(304), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def run(weights, val, data_path, epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_set = MyData(data_path, val, transform=transform)
    train_loader = DataLoader(train_set, batch_size=13, shuffle=True)
    val_set = MyData(data_path, val, transform=transform, is_train=False)
    val_loader = DataLoader(val_set, batch_size=12, shuffle=True)
    net = model.resnet34(num_classes=1).to(device)
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('load weights success')

    opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.05)
    temp_acc = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model=net,
                                                optimizer=opt,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        val_acc = val_one_epoch(model=net,
                                data_loader=val_loader,
                                device=device)

        print(f'epoch {val}-{epoch+1}, train_acc = {train_acc}, val_acc = {val_acc}')

        acc = train_acc * 0.4 + val_acc * 0.6
        if temp_acc < acc:
            torch.save(net.state_dict(), f"./weights/res34_{val}.pt")
            print(f'best_acc = {temp_acc}, all_acc = {acc}, save model!')
            temp_acc = acc
        else:
            print(f'best_acc = {temp_acc}, all_acc = {acc}')


def fold(opt):
    weights, data_path, epochs = opt.weights, opt.data_path, opt.epochs
    for i in range(4):
        run(weights, i+1, data_path, epochs)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=r'')
    parser.add_argument('--data_path', type=str, default=r'E:\Data_for_working\DM_label')
    parser.add_argument('--epochs', type=int, default=50)

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    fold(opt)
