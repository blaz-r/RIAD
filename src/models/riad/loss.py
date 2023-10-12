from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure()

    def forward(self, img: Tensor, img_r: Tensor) -> Tensor:
        return 1 - self.ssim(img, img_r).mean()


class MSGMSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_scales = 4
        self.create_prewitt_kernel()

    def forward(self, img: Tensor, img_r: Tensor) -> Tensor:
        msgms = 0
        # multiscale, first one is non rescaled, then additional halved
        for scale in range(self.num_scales):
            msgms += 1 - self.gms(img, img_r).mean()

            img = F.avg_pool2d(img, kernel_size=2, stride=2)
            img_r = F.avg_pool2d(img_r, kernel_size=2, stride=2)

        return msgms / self.num_scales

    def gms(self, img: Tensor, img_r: Tensor) -> Tensor:
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

    def create_prewitt_kernel(self) -> tuple[Tensor, Tensor]:
        # (1, 1, 3, 3)
        prewitt_x = torch.Tensor([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]]) / 3.0
        # (out_c, in_c, h, w) - (1, 3, 3, 3)
        self.prewitt_x = prewitt_x.repeat(1, 3, 1, 1)
        prewitt_y = torch.Tensor([[[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]]]) / 3.0
        self.prewitt_y = prewitt_y.repeat(1, 3, 1, 1)


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
    loss = RIADLoss()

    x = torch.rand(4, 3, 50, 50)

    print(loss(x, x * 0.75))
