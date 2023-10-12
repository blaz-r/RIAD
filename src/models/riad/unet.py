import torch.nn as nn
import torch


class UNetBlockBasic(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlockBasic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)

        self.relu = nn.functional.relu

    def forward(self, x):
        c1 = self.conv1(x)
        a1 = self.relu(c1)

        c2 = self.conv2(a1)
        a2 = self.relu(c2)

        return a2


class UNetBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlockDown, self).__init__()
        self.u_basic = UNetBlockBasic(in_channels, out_channels)
        self.max_p = nn.MaxPool2d(2, 2)

        self.skip = torch.NoneType

    def forward(self, x):
        c = self.u_basic(x)

        self.skip = c

        down = self.max_p(c)

        return down


class UNetBlockConn(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(UNetBlockConn, self).__init__()
        self.u_basic = UNetBlockBasic(in_channels, mid_channels)
        self.up_conv = nn.ConvTranspose2d(mid_channels, out_channels, 2, stride=2, bias=False)

    def forward(self, x):
        c = self.u_basic(x)

        up = self.up_conv(c)

        return up


class UNetBlockUp(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(UNetBlockUp, self).__init__()
        self.u_basic = UNetBlockBasic(in_channels, mid_channels)
        self.up_conv = nn.ConvTranspose2d(mid_channels, out_channels, 2, stride=2, bias=False)

    def forward(self, x, skip):
        skip_cat = torch.concat((skip, x), 1)

        c = self.u_basic(skip_cat)

        up = self.up_conv(c)

        return up


class UNetBlockOut(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(UNetBlockOut, self).__init__()
        self.u_basic = UNetBlockBasic(in_channels, mid_channels)
        self.out_c = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x, skip):
        skip_cat = torch.concat((skip, x), 1)

        c = self.u_basic(skip_cat)

        out = self.out_c(c)

        return out


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        self.down1 = UNetBlockDown(in_channels, 64)
        self.down2 = UNetBlockDown(64, 128)
        self.down3 = UNetBlockDown(128, 256)
        self.down4 = UNetBlockDown(256, 512)

        self.conn_c = UNetBlockConn(512, 1024, 512)
        self.up1 = UNetBlockUp(1024, 512, 256)
        self.up2 = UNetBlockUp(512, 256, 128)
        self.up3 = UNetBlockUp(256, 128, 64)
        self.out = UNetBlockOut(128, 64, out_channels)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        conn = self.conn_c(d4)
        u1 = self.up1(conn, self.down4.skip)
        u2 = self.up2(u1, self.down3.skip)
        u3 = self.up3(u2, self.down2.skip)
        out = self.out(u3, self.down1.skip)

        return out
