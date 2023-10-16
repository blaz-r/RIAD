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
            root="../datasets/mvtec",
            category="cable",
            image_size=(256, 256),
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            seed=42
        )
        self.datamodule.setup()

        self.unet = UNet(in_channels=3, out_channels=3, transpose=False)
        self.loss = RIADLoss()
        self.msgms = MSGMS()

        self.cutout_sizes = [2, 4, 8, 16]
        self.num_masks = 3
        self.mean_kernel = nn.Parameter(torch.ones(1, 1, 21, 21) / 21 ** 2)

        self.optimizer = torch.optim.AdamW(params=self.unet.parameters(), lr=0.0001)
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
            anomaly_map += self.mean_smoothing(1 - msgms)

        # final map is avg of maps at all coutout sizes
        anomaly_map /= len(self.cutout_sizes)

        return anomaly_map

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
        torch.save(self.state_dict(), f"models/riad/checkpoints/model_{name}.pth")

    def load_model(self, name):
        print("Loading model")
        self.load_state_dict(torch.load(f"models/riad/checkpoints/{name}.pth"))

    def train_riad(self, device: torch.device):
        self.train()
        self.to(device)

        best_p_auroc, best_i_auroc = 0, 0

        epochs = 300
        for epoch in range(epochs):
            train_loader = self.datamodule.train_dataloader()
            total_loss = 0
            with tqdm(total=len(train_loader), desc=str(epoch) + "/" + str(epochs), miniters=int(1),
                      unit='batch') as prog_bar:
                for i, batch in enumerate(train_loader):
                    self.optimizer.zero_grad()

                    image_batch = batch["image"].to(device)
                    reconstructed = self.forward(image_batch)

                    loss = self.loss(image_batch, reconstructed)
                    total_loss += loss.detach().cpu().item()
                    loss.backward()
                    self.optimizer.step()

                    prog_bar.set_postfix(**{"batch_loss": np.round(loss.data.cpu().detach().numpy(), 5),
                                            "avg_loss": np.round(total_loss / (i + 1), 5)})
                    prog_bar.update(1)

            self.scheduler.step()

            if epoch % 20 == 0:
                i_auroc, p_auroc = self.test_riad(device)
                if i_auroc > best_i_auroc or p_auroc > best_p_auroc:
                    best_p_auroc = max(best_p_auroc, p_auroc)
                    best_i_auroc = max(best_i_auroc, i_auroc)
                    self.save_model(f"{p_auroc*1000:.0f}p_{i_auroc*1000:.0f}i")
                self.train()

        self.save_model("last")

    def test_riad(self, device):
        self.eval()
        self.to(device)

        img_auroc, pixel_auroc = AUROC().cpu(), AUROC().cpu()

        test_loader = self.datamodule.test_dataloader()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                image_batch = batch["image"].to(device)
                anomaly_map = self.forward(image_batch).detach().cpu()

                norm_anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
                pixel_auroc.update(norm_anomaly_map, batch["mask"].detach().cpu())
                img_auroc.update(norm_anomaly_map.reshape(norm_anomaly_map.shape[0], -1).max(dim=1).values, batch["label"].detach().cpu())

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
