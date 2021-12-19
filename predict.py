import os
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import ResNet


@torch.no_grad()
def predict(image_path: str, weights: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert os.path.exists(image_path), f"file: '{image_path}' dose not exist."
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize(400), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img = transform(img)
    img = torch.unsqueeze(img, dim=0)

    model = ResNet().to(device)
    assert os.path.exists(weights), f"load weights {weights} failed"
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    output = model(img.to(device))
    print(torch.sigmoid(output))


if __name__ == '__main__':
    pass