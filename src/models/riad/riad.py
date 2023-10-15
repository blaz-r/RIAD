import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn as nn

import torch.nn.functional as F
from anomalib.data.mvtec import MVTec
from anomalib.utils.metrics import AUROC

from tqdm import tqdm

from loss import RIADLoss, MSGMS
from unet import UNet


class RIAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.datamodule = MVTec(
            root="../../../datasets/mvtec",
            category="bottle",
            image_size=(256, 256),
            train_batch_size=4,
            eval_batch_size=1,
            num_workers=0,
        )
        self.datamodule.setup()

        self.unet = UNet(in_channels=3, out_channels=3)
        self.loss = RIADLoss()
        self.msgms = MSGMS()

        self.cutout_sizes = [2, 4, 8, 16]
        self.num_masks = 3
        self.mean_kernel = nn.Parameter(torch.ones(1, 1, 21, 21) / 21 ** 2)

        self.optimizer = torch.optim.AdamW(params=self.unet.parameters() ,lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                         step_size=250, gamma=0.1)

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            cutout_size = random.choice(self.cutout_sizes)
            return self.reconstruct(input, cutout_size)

        b, _, h, w = input.shape
        # inference
        anomaly_map = torch.zeros(b, 1, h, w, device=input.device)
        # reconstruct for all cutout sizes
        for cutout_size in self.cutout_sizes:
            reconstructed = self.reconstruct(input, cutout_size)
            msgms = self.msgms(input, reconstructed, as_map=True)
            # get anomaly map at current scale by subtracting smoothed msgms from 1
            anomaly_map += 1 - self.mean_smoothing(msgms)

        # final map is avg of maps at all coutout sizes
        anomaly_map /= len(self.cutout_sizes)

        return {"anomaly_map": anomaly_map, "anomaly_score": torch.max(anomaly_map)}

    def reconstruct(self, input: Tensor, cutout_size: int):
        disjoint_masks = self.create_masks(input.shape, cutout_size, input.device)

        reconstruction = torch.zeros_like(input)
        for mask in disjoint_masks:
            masked = mask * input
            inpainted = self.unet(masked)
            reconstruction += inpainted * (1 - mask)

        return reconstruction

    def create_masks(self, img_size: torch.Size, cutout_size: int, device: str) -> list[Tensor]:
        b, _, h, w = img_size
        assert h % cutout_size == 0, "image size must be divisible by cutout size"
        assert w % cutout_size == 0, "image size must be divisible by cutout size"

        grid_h, grid_w = h // cutout_size, w // cutout_size
        num_cells = grid_w * grid_h
        random_indices = torch.randperm(num_cells)

        disjoint_masks = []
        # split random pixels into num_masks chunks
        for indices in torch.chunk(random_indices, self.num_masks):
            # flattened mask where selected indices are masked by 0
            mask = torch.ones(num_cells, requires_grad=False, device=device)
            mask[indices] = 0
            # reshape back into right shape
            mask = mask.reshape(grid_h, grid_w)
            # repeat in each dim to make mask back into h * w, and each cutout properly sized
            mask = torch.repeat_interleave(mask, cutout_size, dim=0)
            mask = torch.repeat_interleave(mask, cutout_size, dim=1)
            disjoint_masks.append(mask)

        return disjoint_masks

    def mean_smoothing(self, input_map: Tensor) -> Tensor:
        return F.conv2d(input_map, self.mean_kernel, padding=21 // 2)

    def save_model(self, name: str):
        print("Saving model")
        torch.save(self.state_dict(), f"checkpoints/model_{name}.pth")

    def load_model(self):
        print("Loading model")
        self.load_state_dict(torch.load("checkpoints/model.pth"))

    def train_riad(self, device: torch.device):
        self.train()
        self.to(device)

        best_auroc = 0

        epochs = 300
        for epoch in range(epochs):
            train_loader = self.datamodule.train_dataloader()
            with tqdm(total=len(train_loader), desc=str(epoch) + "/" + str(epochs), miniters=int(1),
                      unit='batch') as prog_bar:
                for batch in train_loader:
                    self.optimizer.zero_grad()

                    image_batch = batch["image"].to(device)
                    reconstructed = self.forward(image_batch)

                    loss = self.loss(image_batch, reconstructed)
                    loss.backward()
                    self.optimizer.step()

                    prog_bar.set_postfix(**{'loss': np.round(loss.data.cpu().detach().numpy(), 5)})
                    prog_bar.update(1)

            self.scheduler.step()

            if epoch % 50 == 0:
                i_auroc, p_auroc = self.test_riad(device)
                avg_auroc = (i_auroc + p_auroc) / 2
                if avg_auroc > best_auroc:
                    best_auroc = avg_auroc
                    self.save_model(f"{avg_auroc*1000:.0f}")
                self.train()

        self.save_model("last")

    def test_riad(self, device):
        self.eval()
        self.to(device)

        img_auroc, pixel_auroc = AUROC().cpu(), AUROC().cpu()

        test_loader = self.datamodule.test_dataloader()
        for batch in tqdm(test_loader):
            image_batch = batch["image"].to(device)
            results = self.forward(image_batch)

            img_auroc.update(results["anomaly_score"].detach().cpu(), batch["label"].detach().cpu())
            anomaly_map = results["anomaly_map"].detach().cpu()
            norm_anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
            pixel_auroc.update(norm_anomaly_map, batch["mask"].detach().cpu())

        i_auroc_val = img_auroc.compute().item()
        p_auroc_val = pixel_auroc.compute().item()

        print(f"Image AUROC: {i_auroc_val}")
        print(f"Pixel AUROC: {p_auroc_val}")

        return i_auroc_val, p_auroc_val


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RIAD()

    model.train_riad(device)
    model.test_riad(device)
