import os

from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class MyData(Dataset):

    def __init__(self, path: str, val=1, is_train=True, transform=None):
        self.train_data_set = list()
        del self.train_data_set[:]
        self.val_data_set = list()
        del self.val_data_set[:]
        self.transform = transform
        self.is_train = is_train
        if is_train:
            for i in range(4):
                if i+1 == val:
                    continue
                self.path = path + '/' + f'{i+1}'
                for image_name in os.listdir(self.path):
                    image_path = self.path + '/' + image_name
                    self.train_data_set.append(image_path)
        else:
            self.path = path + '/' + f'{val}'
            for image_name in os.listdir(self.path):
                image_path = self.path + '/' + image_name
                self.val_data_set.append(image_path)

    def __len__(self):
        if self.is_train:
            return len(self.train_data_set)
        else:
            return len(self.val_data_set)

    def __getitem__(self, item):
        if self.is_train:
            image = self.train_data_set[item]
            image = Image.open(image).convert('RGB')
            label = self.train_data_set[item][-5:-4]

            if self.transform is not None:
                image = self.transform(image)

            return image, np.float32(label)

        else:
            image = self.val_data_set[item]
            image = Image.open(image).convert('RGB')
            label = self.val_data_set[item][-5:-4]

            if self.transform is not None:
                image = self.transform(image)

            return image, np.float32(label)
