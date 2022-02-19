import os

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt


class MyData(Dataset):
    data_set = list()

    def __init__(self, path: str, val=1, is_train=True, transform=None):
        self.transform = transform
        if is_train:
            for i in range(4):
                if i+1 == val:
                    continue
                self.path = path + '/' + f'{i+1}'
                for image_name in os.listdir(self.path):
                    image_path = self.path + '/' + image_name
                    self.data_set.append(image_path)
        else:
            self.path = path + '/' + f'{val}'
            for image_name in os.listdir(self.path):
                image_path = self.path + '/' + image_name
                self.data_set.append(image_path)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        image = self.data_set[item]
        # Image.open(image).show()
        image = Image.open(image).convert('RGB')
        label = self.data_set[item][-5:-4]

        if self.transform is not None:
            image = self.transform(image)

        return image, np.float32(label)


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    data = MyData(r'E:/Data_for_working/DM_label', is_train=True)
    data1 = MyData(r'E:/Data_for_working/DM_label', is_train=False)