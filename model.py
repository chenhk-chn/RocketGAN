import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Encoder(nn.Module):
    """Encoder network."""

    def __init__(self, conv_dim=64, repeat_num=6):
        super(Encoder, self).__init__()

        layers_1 = [nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
                    nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True), nn.ReLU(inplace=True)]
        self.layers_1 = nn.Sequential(*layers_1)

        # Down-sampling layers.
        curr_dim = conv_dim
        layers_2 = [nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True), nn.ReLU(inplace=True)]
        self.layers_2 = nn.Sequential(*layers_2)
        curr_dim = curr_dim * 2

        layers_3 = [nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True), nn.ReLU(inplace=True)]
        self.layers_3 = nn.Sequential(*layers_3)
        curr_dim = curr_dim * 2

        # Bottleneck layers.
        res_block = []
        for i in range(repeat_num):
            res_block.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        x = self.layers_1(x)
        x = self.layers_2(x)
        x = self.layers_3(x)
        out = self.res_block(x)

        return out  # [16,128,128,64]


class Decoder(nn.Module):
    """Decoder network."""

    def __init__(self, curr_dim=256, c_dim=5):
        super(Decoder, self).__init__()

        # up-sample Layer1
        layers_1 = [
            nn.ConvTranspose2d(curr_dim + c_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True), nn.ReLU(inplace=True)]
        curr_dim = curr_dim // 2
        self.layers_1 = nn.Sequential(*layers_1)

        # up-sample Layer2
        layers_2 = [nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True), nn.ReLU(inplace=True)]
        curr_dim = curr_dim // 2
        self.layers_2 = nn.Sequential(*layers_2)

        # up-sample Layer3
        layers_3 = [nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False), nn.Tanh()]
        self.layers_3 = nn.Sequential(*layers_3)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        x = self.layers_1(x)
        x = self.layers_2(x)
        out = self.layers_3(x)
        return out


class Discriminator(nn.Module):
    """Discriminator Network."""

    def __init__(self, c_dim):
        super(Discriminator, self).__init__()
        curr_dim = 32
        kernel_size = 3
        stride = 2

        layer = [nn.ReflectionPad2d(1),
                 spectral_norm(nn.Conv2d(3, curr_dim, kernel_size=kernel_size, stride=stride)), nn.LeakyReLU(0.2)]

        for _ in range(4):
            layer += [nn.ReflectionPad2d(1),
                      spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=kernel_size, stride=stride)),
                      nn.BatchNorm2d(curr_dim * 2), nn.LeakyReLU(0.2)]
            curr_dim = curr_dim * 2
        self.main = nn.Sequential(*layer)

        self.adv = nn.Sequential(spectral_norm(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=2)))
        cls = [spectral_norm(nn.Conv2d(curr_dim, c_dim, kernel_size=3, stride=1)), nn.AdaptiveAvgPool2d(1)]
        self.cls = nn.Sequential(*cls)

    def forward(self, x):
        out = self.main(x)
        adv = self.adv(out)
        cls = self.cls(out)
        return adv, cls.squeeze(2).squeeze(2)
