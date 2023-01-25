import os
import torch
import torchvision.transforms as transforms

from model import resnet34, efficientnetv2_s, mobilenet_v3_small
from dataset import MyData
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(image_path: str, weights1: str, weights2: str, weights3: str, weights4: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_list = [weights1, weights2, weights3, weights4]
    assert os.path.exists(image_path), f"file: '{image_path}' dose not exist."

    transform = transforms.Compose(
        [transforms.Resize(304), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_data = MyData(image_path, val=5, is_train=False, transform=transform)
    test_loader = DataLoader(test_data, batch_size=24, shuffle=False)

    all_class_pred = torch.zeros([24]).to(device)
    sample_num = 24
    accu_num1 = torch.zeros(1).to(device)

    for weights in weight_list:
        model = resnet34(num_classes=1).to(device)
        model.load_state_dict(torch.load(weights))
        print(f'load {weights} success')

        accu_temp = torch.zeros(1).to(device)
        for step, data in enumerate(test_loader):
            images, labels = data
            # sample_num += images.shape[0]
            pred = model(images.to(device))
            class_pred = torch.tensor(torch.sigmoid(pred))

            class_pred[torch.where(class_pred >= 0.2)] = 1
            class_pred[torch.where(class_pred < 0.2)] = 0
            accu_temp += torch.eq(class_pred, labels.to(device)).sum()
            # print(class_pred, accu_temp.item() / sample_num)
            all_class_pred += class_pred

    print(all_class_pred)
    accu_num1 += torch.eq(all_class_pred, labels.to(device) * 2 + 1).sum()
    accu_num1 += torch.eq(all_class_pred, labels.to(device) * 2).sum()
    print(labels, accu_num1.item() / sample_num)


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    predict(r'/home/lee/Work/Pycharmprojects/pytorch_resnet/DM_label', r'weights/res34_1_100epoch.pt',
            r'weights/res34_2_100epoch.pt', r'weights/res34_3_100epoch.pt', r'weights/res34_4_100epoch.pt')
