from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from torchvision.transforms.v2 import Grayscale


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure()

    def forward(self, img: Tensor, img_r: Tensor) -> Tensor:
        return 1 - self.ssim(img_r, img)


class MSGMS(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_scales = 4

        self.prewitt_x = nn.Parameter(torch.Tensor([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]]) / 3.0)
        self.prewitt_y = nn.Parameter(torch.Tensor([[[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]]]) / 3.0)
        self.to_gray = Grayscale()

    def forward(self, img: Tensor, img_r: Tensor, as_map: bool = False) -> Tensor:
        b, _, h, w = img.shape
        if as_map:
            msgsm = torch.zeros(b, 1, h, w, device=img.device)
        else:
            msgsm = 0

        # multiscale, first one is non rescaled, following ones are halved
        for scale in range(self.num_scales):
            gms = self.gms(img, img_r)
            if as_map:
                msgsm += F.interpolate(gms, size=(h, w), mode="bilinear", align_corners=False)
            else:
                msgsm += torch.mean(gms)

            img = F.avg_pool2d(img, kernel_size=2, stride=2)
            img_r = F.avg_pool2d(img_r, kernel_size=2, stride=2)

        # if map, return per pixel avg of all scales, else return total average
        return msgsm / self.num_scales

    def gms(self, img: Tensor, img_r: Tensor) -> Tensor:
        img = self.to_gray(img)
        img_r = self.to_gray(img_r)

        gi_x = F.conv2d(img, self.prewitt_x, stride=1, padding=1)
        gi_y = F.conv2d(img, self.prewitt_y, stride=1, padding=1)
        gi = torch.sqrt(gi_x**2 + gi_y**2)

        gir_x = F.conv2d(img_r, self.prewitt_x, stride=1, padding=1)
        gir_y = F.conv2d(img_r, self.prewitt_y, stride=1, padding=1)
        gir = torch.sqrt(gir_x**2 + gir_y**2)

        # Constant c from https://arxiv.org/pdf/1308.3052.pdf
        c = 0.0026

        gsm = (2 * gi * gir + c) / (gi**2 + gir**2 + c)
        return gsm


class MSGMSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_scales = 4

        self.msgms = MSGMS()

    def forward(self, img: Tensor, img_r: Tensor) -> Tensor:
        msgms_value = self.msgms(img, img_r)

        # 1 can be moved out of summation, where msgms is mean(mean(gms)) for all gms scales
        return 1 - msgms_value


class RIADLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim_loss = SSIMLoss()
        self.msgms_loss = MSGMSLoss()
        self.lambda_g, self.lambda_s = 1, 1

    def forward(self, img: Tensor, img_r: Tensor) -> Tensor:
        lg = self.msgms_loss(img, img_r)
        ls = self.ssim_loss(img, img_r)
        l2 = F.mse_loss(img, img_r)

        return self.lambda_g * lg + self.lambda_s * ls + l2


if __name__ == "__main__":
    from kornia.losses import ssim_loss
    loss = SSIMLoss()

    x = torch.rand(4, 3, 50, 50)
    y = torch.rand(4, 3, 50, 50)

    print(loss(x, y))
    print(ssim_loss(x, y, 11) * 2)

    # msgsm = MSGMS().cuda()
    # map = msgsm(x.cuda(), x.cuda(), as_map=True)
    # print(map.shape)

    msgms = MSGMSLoss()
    print(msgms(x, y))
