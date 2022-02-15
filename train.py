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

transform = transforms.Compose(
    [transforms.Resize(304), transforms.ToTensor()])


def run(opt):
    weights, train_path, val_path, batch_size, epochs = opt.weights, opt.train_path, opt.val_path, opt.batch_size, opt.epochs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_set = MyData(train_path, transform=transform)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_set = MyData(val_path, transform=transform)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    net = ResNet().to(device)
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('load weights success')

    opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5E-5)
    save_loss = float('inf')
    temp_acc = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model=net,
                                                optimizer=opt,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        print(f'epoch {epoch}, train_loss = {train_loss}, acc = {train_acc}')

        val_acc = val_one_epoch(model=net,
                                data_loader=val_loader,
                                device=device)

        if temp_acc < val_acc:
            torch.save(net.state_dict(), "./weights/2_15best.pt")
            print(f'best_acc = {temp_acc}, val_acc = {val_acc}, save model!')
            temp_acc = val_acc
        else:
            print(f'best_acc = {temp_acc}, val_acc = {val_acc}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=' ')
    parser.add_argument('--train_path', type=str, default=r'/home/lee/Work/data/DM_label (copy)')
    parser.add_argument('--val_path', type=str, default=r'/home/lee/Work/data/DM_label_val')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    run(opt)
