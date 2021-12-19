import argparse
import os
import parser
import torch

from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import ResNet
from dataset import MyData
from utils import train_one_epoch, val_one_epoch

transform = transforms.Compose([transforms.Resize(400), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def run(opt):
    weights, path, batch_size, epoch = opt.weihts, opt.data_path, opt.batch_size, opt.epoch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_set = MyData(path, transform=transform)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_set = MyData(path, transform=transform)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    net = ResNet().to(device)
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('load weights success')

    opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5E-5)
    save_loss = float('inf')
    for epoch in range(300):
        train_loss, train_acc = train_one_epoch(model=net,
                                                optimizer=opt,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        if train_loss < save_loss:
            # torch.save(net.state_dict(), "./weights/mob_best.pt")
            print(f'epoch {epoch} save weights success, train_loss = {train_loss}')
            save_loss = train_loss

        val_acc = val_one_epoch(model=net,
                                data_loader=val_loader,
                                device=device)

        print(f'val_acc = {val_acc}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/mob_best.pt')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=300)

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    run(opt)
