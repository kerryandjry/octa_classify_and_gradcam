import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from model import resnet34, efficientnetv2_s, mobilenet_v3_small
from torchvision import transforms
from grad_cam import GradCAM, show_cam_on_image, center_crop_img


device = 'cuda' if torch.cuda.is_available() else 'cpu'
def main():
    model = resnet34(num_classes=1).to(device)
    weights = r'weights/res_1_100epoch.pt'
    model.load_state_dict(torch.load(weights), map_location=torch.device(device))
    print('load weights success')
    target_layers = [model.feature[-1]]

    # target_layers = [model.features]
    # target_layers = [model.layer4]

    data_transform = transforms.Compose(
        [transforms.Resize(304), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    img_path = "DM_label/5/2_1.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 0  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
