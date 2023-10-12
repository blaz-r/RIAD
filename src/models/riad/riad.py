import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn as nn

from anomalib.data.mvtec import MVTec
from tqdm import tqdm

from loss import RIADLoss
from unet import UNet


class RIAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = RIADLoss()
        self.datamodule = MVTec(
            root="../../../datasets/mvtec",
            category="bottle",
            image_size=(256, 256),
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
        )
        self.datamodule.setup()
        self.unet = UNet(in_channels=3, out_channels=3)

        self.cutout_sizes = [2, 4, 8, 16]
        self.num_masks = 3

        self.optimizer = torch.optim.AdamW(params=self.unet.parameters() ,lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                         step_size=250, gamma=0.1)

    def forward(self, input: Tensor) -> Tensor:
        return self.reconstruct(input)

    def reconstruct(self, input: Tensor):

        cutout_size = random.choice(self.cutout_sizes)
        disjoint_masks = self.create_masks(input.shape, cutout_size)

        reconstruction = torch.zeros_like(input)
        for mask in disjoint_masks:
            masked = mask * input
            inpainted = self.unet(masked)
            reconstruction += inpainted * (1 - mask)

        return reconstruction

    def create_masks(self, img_size: torch.Size, cutout_size: int) -> list[Tensor]:
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
            mask = torch.ones(num_cells, requires_grad=False)
            mask[indices] = 0
            # reshape back into right shape
            mask = mask.reshape(grid_h, grid_w)
            # repeat in each dim to make mask back into h * w, and each cutout properly sized
            mask = torch.repeat_interleave(mask, cutout_size, dim=0)
            mask = torch.repeat_interleave(mask, cutout_size, dim=1)
            disjoint_masks.append(mask)

        return disjoint_masks

    def train_riad(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.unet.train()
        self.unet.to(device)

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


if __name__ == "__main__":
    x = torch.rand(4, 3, 32, 32)

    model = RIAD()

    model.train_riad()
