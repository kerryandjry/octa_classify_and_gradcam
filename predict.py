import os
import torch
import torchvision.transforms as transforms

from model import resnet34, efficientnetv2_s, mobilenet_v3_small
from dataset import MyData
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(image_path: str, weights: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert os.path.exists(image_path), f"file: '{image_path}' dose not exist."
    accu_num = torch.zeros(1).to(device)

    transform = transforms.Compose([transforms.Resize(304), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_data = MyData(image_path, val=5, is_train=False, transform=transform)
    test_loader = DataLoader(test_data, batch_size=87, shuffle=False)

    model = efficientnetv2_s(num_classes=1).to(device)
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights))
        print('load weights success')
    assert os.path.exists(weights), f"load weights {weights} failed"

    sample_num = 0

    for step, data in enumerate(test_loader):
        images, labels = data

        sample_num += images.shape[0]
        pred = model(images.to(device))
        class_pred = torch.tensor(torch.sigmoid(pred))
        class_pred[torch.where(class_pred >= 0.37)] = 1
        class_pred[torch.where(class_pred < 0.37)] = 0
        print(class_pred, "\n", labels)
        accu_num += torch.eq(class_pred, labels.to(device)).sum()
    return accu_num.item() / sample_num


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print(predict(r'/home/lee/Work/Pycharmprojects/pytorch_resnet/DM_label', r'weights/eff_4_100epoch.pt'))
