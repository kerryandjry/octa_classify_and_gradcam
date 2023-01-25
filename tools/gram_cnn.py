import os
import pathlib
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from model import resnet34, efficientnetv2_s, mobilenet_v3_small
from torchvision import transforms
from grad_cam import GradCAM, show_cam_on_image, center_crop_img

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    model = resnet34(num_classes=1)
    weights = r'weights/res_1_100epoch.pt'
    model.load_state_dict(torch.load(weights))
    print('load weights success')
    target_layers = [model.layer4]
    # target_layers = [model.features[-1]]

    data_transform = transforms.Compose(
        [transforms.Resize(304), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    i = 272
    # load image
    for img_path in pathlib.Path('../DM_label/5/').iterdir():
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path).convert('RGB')

        # [C, H, W]
        tran_img = data_transform(img)
        # expand batch dimension
        # [C, H, W] -> [N, C, H, W]
        input_tensor = torch.unsqueeze(tran_img, dim=0)

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        target_category = int(str(img_path)[-5:-4])  # tabby, tabby cat
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        grayscale_cam = grayscale_cam[0, :]
        if np.sum(grayscale_cam) == 0:
            continue
        print(f"{img_path}, label={target_category}")
        img = np.array(img, dtype=np.uint8)
        img.resize((304, 304, 3))

        img.astype(dtype=np.float32)
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                          grayscale_cam,
                                          use_rgb=True)
        new_img = np.concatenate((img, visualization), axis=1)
        plt.imshow(new_img)
        plt.savefig(f"grad_cam_img_layer4/{i}.jpg")
        i += 1


if __name__ == '__main__':
    main()
