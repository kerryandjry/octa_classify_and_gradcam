import os
import torch
import torchvision.transforms as transforms

from model import resnet34, efficientnetv2_s, mobilenet_v3_small
from dataset import MyData
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(image_path: str, weights1: str, weights2: str, weights3: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert os.path.exists(image_path), f"file: '{image_path}' dose not exist."
    accu_num1 = torch.zeros(1).to(device)

    transform = transforms.Compose([transforms.Resize(304), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_data = MyData(image_path, val=5, is_train=False, transform=transform)
    test_loader = DataLoader(test_data, batch_size=87, shuffle=False)

    model1 = efficientnetv2_s(num_classes=1).to(device)
    model2 = resnet34(num_classes=1).to(device)
    model3 = mobilenet_v3_small(num_classes=1).to(device)
    if os.path.exists(weights1) and os.path.exists(weights2) and os.path.exists(weights3):
        model1.load_state_dict(torch.load(weights1))
        model2.load_state_dict(torch.load(weights2))
        model3.load_state_dict(torch.load(weights3))
        # print('load weights success')
    assert os.path.exists(weights1), f"load weights failed"

    sample_num = 0

    for step, data in enumerate(test_loader):
        images, labels = data

        sample_num += images.shape[0]
        pred1 = model1(images.to(device))
        pred2 = model2(images.to(device))
        pred3 = model3(images.to(device))
        class_pred1 = torch.tensor(torch.sigmoid(pred1))
        class_pred2 = torch.tensor(torch.sigmoid(pred2))
        class_pred3 = torch.tensor(torch.sigmoid(pred3))
        class_pred1[torch.where(class_pred1 >= 0.37)] = 1
        class_pred1[torch.where(class_pred1 < 0.37)] = 0
        class_pred2[torch.where(class_pred2 >= 0.37)] = 1
        class_pred2[torch.where(class_pred2 < 0.37)] = 0
        class_pred3[torch.where(class_pred3 >= 0.37)] = 1
        class_pred3[torch.where(class_pred3 < 0.37)] = 0
        class_pred = class_pred1 + class_pred2 + class_pred3
        class_pred = torch.where(class_pred > 1, 1., 0.)
        # print(class_pred)
        accu_num1 += torch.eq(class_pred, labels.to(device)).sum()
        print(accu_num1.item() / sample_num)


if __name__ == '__main__':
    test = 4
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    predict(r'/home/lee/Work/Pycharmprojects/pytorch_resnet/DM_label', rf'weights/eff_{test}_100epoch.pt', rf'weights/res_{test}_100epoch.pt', rf'weights/mobile_{test}_100epoch.pt')
    for _ in range(100):
        predict(r'/home/lee/Work/Pycharmprojects/pytorch_resnet/DM_label', rf'weights/eff_{test}_100epoch.pt',
                rf'weights/res_{test}_100epoch.pt', rf'weights/mobile_{test}_100epoch.pt')

