import torch
from torch import nn
from PIL import Image
from typing import Union, List

from model import resnet34, efficientnetv2_s, mobilenet_v3_small

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VectorFusion(nn.Module):
    def __init__(self, res_weight: str, eff_weight: str, mob_weight: str, img: Union[str, List[str]]):
        super().__init__()
        self.img = Image.open(img).convert('RGB')
        self.res_model = resnet34(num_classes=1).to(device)
        self.eff_model = efficientnetv2_s(num_classes=1).to(device)
        self.mob_model = mobilenet_v3_small(num_classes=1).to(device)
        self.res_model.load_state_dict(torch.load(res_weight))
        self.eff_model.load_state_dict(torch.load(eff_weight))
        self.mob_model.load_state_dict(torch.load(mob_weight))

        self.reduce_dim = nn.Conv1d(in_channels=304, out_channels=128, kernel_size=1)
        self.reduce_dim2 = nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1)
        self.out = nn.Linear(128 * 3, 2)

    #         self.w1 = nn.Parameter(torch.zeros(1))
    #         self.w2 = nn.Parameter(torch.zeros(1))
    #         self.w3 = nn.Parameter(torch.zeros(1))

    def forward(self):

        res_vec = self.res_model(self.img.to(device))
        res_emb = res_vec.future.view(res_vec.shape[0], res_vec.shape[1], 1)
        mob_vec = self.mob_model(self.img.to(device))
        mob_emb = mob_vec.future.view(mob_vec.shape[0], mob_vec.shape[1], 1)
        eff_vec = self.eff_model(self.to(device))
        eff_emb = eff_vec.future.view(eff_vec.shape[0], eff_vec.shape[1], 1)

        # img_emb = self.img_model(img)
        # img_emb = img_emb.view(img_emb.shape[0], img_emb.shape[1], 1)
        # img_emb = self.reduce_dim(img_emb)
        # img_emb = img_emb.view(img_emb.shape[0], img_emb.shape[1])

        # summing up the vectors
        # text_emb = cam_emb[0] + flau_emb[0]

        # Bilinear
        # text_emb = text_emb.view(text_emb.shape[0],1,text_emb.shape[1])

        # Bilinear Pooling
        # pool_emb = torch.bmm(img_emb,text_emb)
        # pool_emb = self.reduce_dim2(pool_emb).view(text_emb.shape[0],768)
        fuse = torch.cat([res_emb, mob_emb, eff_emb])
        logits = self.out(fuse)
        return logits


if __name__ == '__main__':
   pred = VectorFusion("weights/res34_4_100epoch.pt", "weights/effi_s_4_100epoch.pt", "weights/mobile_4_100epoch.pt", r"DM_label/2/_51470F_51470F__1633_Angio Retina_OS_2018-12-13_09-23-04_F_1962-06-03_Enface-304x304-Superficial_0.png")
