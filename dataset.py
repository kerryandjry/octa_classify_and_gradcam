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
        # Image.open(image).show()
        image = Image.open(image).convert('RGB')
        plt.imshow(image)
        label = self.data_set[item][-5:-4]

        if self.transform is not None:
            image = self.transform(image)

        return image, np.float32(label)


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    transform = transforms.Compose([transforms.Resize(304), transforms.ToTensor()])
    data = MyData(r'/home/lee/Work/data/DM_label (copy)', transform=transform)
    print(data[2])