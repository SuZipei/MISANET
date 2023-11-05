import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, batch_size=128, seq_len=384, n_channels=32, n_classes=2):
        super(EEGNet, self).__init__()
        F1, D = 8, 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Layer 1 - Conv2D + BatchNorm
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 128),
                padding=(0, 64),
                bias=False),
            nn.BatchNorm2d(F1)
        )

        # Layer 2 - DepthwiseConv2D + AvgPool2D
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=F1,
                      out_channels=F1,
                      kernel_size=(17, 1),
                      groups=2,
                      bias=False),
            nn.ELU(),
            nn.Conv2d(in_channels=F1,
                      out_channels=F1 * D,
                      kernel_size=(16, 1),
                      groups=2,
                      bias=False),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.5)
        )

        # Layer 3 - SeparableConv2D + AvgPool2D
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=F1 * D,
                      out_channels=F1 * D,
                      kernel_size=(1, 48),
                      padding=(0, 8),
                      groups=16,
                      bias=False),
            nn.Conv2d(in_channels=F1 * D,
                      out_channels=F1 * D,
                      kernel_size=(1, 1),  # Pointwise
                      bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.5)
        )

    def get_feature(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def forward(self, x):
        x = x.reshape(-1, 1, self.n_channels, self.seq_len)
        x = self.get_feature(x)
        x = x.view(x.size(0), -1)
        return x


class EOMGNet(nn.Module):
    def __init__(self, batch_size=128, seq_len=384, n_channels=2, n_classes=2):
        super(EOMGNet, self).__init__()
        F1, D = 8, 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Layer 1 - Conv2D + BatchNorm
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 128),
                padding=(0, 64),
                bias=False),
            nn.BatchNorm2d(F1)
        )

        # Layer 2 - DepthwiseConv2D + AvgPool2D
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=F1,
                      out_channels=F1 * D,
                      kernel_size=(2, 1),
                      groups=2,
                      bias=False),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.5)
        )

        # Layer 3 - SeparableConv2D + AvgPool2D
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=F1 * D,
                      out_channels=F1 * D,
                      kernel_size=(1, 48),
                      padding=(0, 8),
                      groups=16,
                      bias=False),
            nn.Conv2d(in_channels=F1 * D,
                      out_channels=F1 * D,
                      kernel_size=(1, 1),  # Pointwise
                      bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.5)
        )

    def get_feature(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def forward(self, x):
        x = x.reshape(-1, 1, self.n_channels, self.seq_len)
        x = self.get_feature(x)
        x = x.view(x.size(0), -1)
        return x

class EEGNetDecoder(nn.Module):
    def __init__(self, batch_size=128):
        super(EEGNetDecoder, self).__init__()
        F1, D = 8, 2
        self.batch_size = batch_size

        # Inverse of block3
        self.inv_block3 = nn.Sequential(
            nn.ConvTranspose2d(F1 * D, F1 * D, kernel_size=(1, 1)),
            nn.ConvTranspose2d(F1 * D, F1 * D, kernel_size=(1, 48), padding=(0, 8), groups=16)
        )

        # Inverse of block2
        self.inv_block2 = nn.Sequential(
            nn.ConvTranspose2d(F1 * D, F1, kernel_size=(16, 1), groups=2),
            nn.ConvTranspose2d(F1, F1, kernel_size=(17, 1), groups=2),
            nn.Upsample(size=(32, 385), mode='nearest')  # Set target size explicitly
        )

        # Inverse of block1
        self.inv_block1 = nn.Sequential(
            nn.ConvTranspose2d(F1, 1, kernel_size=(1, 128), padding=(0, 64))
        )

    def forward(self, x):
        x = x.view([-1, 16, 1, 8])

        x = self.inv_block3(x)
        x = self.inv_block2(x)
        x = self.inv_block1(x)
        return x


class EOMGNetDecoder(nn.Module):
    def __init__(self, batch_size=128):
        super(EOMGNetDecoder, self).__init__()
        F1, D = 8, 2
        self.batch_size = batch_size

        # Inverse of block3
        self.inv_block3 = nn.Sequential(
            nn.ConvTranspose2d(F1 * D, F1 * D, kernel_size=(1, 1)),
            nn.ConvTranspose2d(F1 * D, F1 * D, kernel_size=(1, 48), padding=(0, 8), groups=16)
        )

        # Inverse of block2
        self.inv_block2 = nn.Sequential(
            nn.ConvTranspose2d(F1 * D, F1, kernel_size=(16, 1), groups=2),
            nn.ConvTranspose2d(F1, F1, kernel_size=(17, 1), groups=2),
            nn.Upsample(size=(2, 385), mode='nearest')  # Set target size explicitly
        )

        # Inverse of block1
        self.inv_block1 = nn.Sequential(
            nn.ConvTranspose2d(F1, 1, kernel_size=(1, 128), padding=(0, 64))
        )

    def forward(self, x):
        x = x.view([-1, 16, 1, 8])

        x = self.inv_block3(x)
        x = self.inv_block2(x)
        x = self.inv_block1(x)
        return x


if __name__ == "__main__":

    net = EEGNet(batch_size=512, seq_len=384, n_channels=32, n_classes=2)
    eognet = EOMGNet(batch_size=512, seq_len=384, n_channels=2, n_classes=2)
    x = torch.rand(512, 1, 32, 384)
    out = net(x)
    print(out.shape)
    inverse_net = EEGNetDecoder(batch_size=512)
    x = inverse_net(out)
    print(x.shape)
