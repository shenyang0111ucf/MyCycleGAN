import torch
import torch.nn as nn
import torch.nn.functional as F

class NewConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, 1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, features=[64, 128, 258, 512]):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(input_channels, features[0],
                                kernel_size=4, stride=2, padding=1, padding_mode="reflect"))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        input_channels = features[0]
        for feature in features[1:]:
            layers.append(NewConvLayer(input_channels, feature, kernel_size=4, stride=1 if feature == features[-1] else 2))
            input_channels = feature
        self.model = nn.Sequential(*layers)

        self.last_conv = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.model(x)
        out = self.last_conv(feat)
        feat = F.avg_pool2d(feat, 4, 1, 0)
        return out, feat

def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(input_channels=3)
    preds, feat = model(x)
    print(preds.shape)
    print(feat.shape)

if __name__ == "__main__":
    test()
