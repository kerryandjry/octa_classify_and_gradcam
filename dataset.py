import os

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


class MyData(Dataset):
    def __init__(self, path: str, transform=None):
        self.path = path
        self.data_set = os.listdir(path)
        self.transform = transform

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        image = self.path + '/' + self.data_set[item]
        img = Image.open(image).convert('RGB')
        label = self.data_set[item][-5:-4]

        if self.transform is not None:
            img = self.transform(img)

        return img, np.float32(label)


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    transform = transforms.Compose([transforms.Resize(400), transforms.ToTensor()])
    # plt.imshow(data[2][0])
    # plt.show()
