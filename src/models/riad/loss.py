from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from torchvision.transforms.v2 import Grayscale
import math


class SSIMLoss(torch.nn.Module):
    """
    Taken from https://github.com/VitjanZ/DRAEM/blob/main/loss.py

    """
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = self.create_window(window_size).cuda()

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = self.create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        s_score, ssim_map = self.ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
        return 1.0 - s_score

    @staticmethod
    def gaussian(window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
        if val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            l = max_val - min_val
        else:
            l = val_range

        padd = window_size // 2
        (_, channel, height, width) = img1.size()

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        c1 = (0.01 * l) ** 2
        c2 = (0.03 * l) ** 2

        v1 = 2.0 * sigma12 + c2
        v2 = sigma1_sq + sigma2_sq + c2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret, ssim_map


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
    my_loss = SSIMLoss()

    x = torch.rand(4, 3, 50, 50)
    y = torch.rand(4, 3, 50, 50)

    print(my_loss(x, y))
    print(ssim_loss(y, x, 11) * 2)

    # msgsm = MSGMS().cuda()
    # map = msgsm(x.cuda(), x.cuda(), as_map=True)
    # print(map.shape)

    msgms = MSGMSLoss()
    print(msgms(x, y))
