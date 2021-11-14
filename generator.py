import torch
import torch.nn as nn


class NewConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, downsample=True, use_act=True, **kwargs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, padding_mode='reflect', **kwargs)
            if downsample
            else nn.ConvTranspose2d(input_channels, output_channels, **kwargs),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.model(x)


class ResidualLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            NewConvLayer(channels, channels, kernel_size=3, padding=1),
            NewConvLayer(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels, num_features = 64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )

        self.down_layer = nn.Sequential(
                NewConvLayer(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
                NewConvLayer(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1),
        )
        self.res_layer = nn.Sequential(
            *[ResidualLayer(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_layer = nn.Sequential(
                NewConvLayer(num_features * 4, num_features * 2, downsample=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1),
                NewConvLayer(num_features * 2, num_features * 1, downsample=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1)
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(num_features * 1, img_channels, kernel_size=7, stride=1, padding=3,
                              padding_mode="reflect"),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down_layer(x)
        x = self.res_layer(x)
        x = self.up_layer(x)
        return self.last_layer(x)
