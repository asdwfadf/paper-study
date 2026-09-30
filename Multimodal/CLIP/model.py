import torch
import torch.nn as nn


class RNBlock(nn.Module):
    def __init__(self, in_c, inner_c, out_c, stride):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_c, inner_c, 1, stride=1, bias=False),
            nn.BatchNorm2d(inner_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(inner_c, inner_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(inner_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(inner_c, out_c, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_c),
        )

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        out = self.relu(shortcut + residual)

        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(2, stride=2)
        )

        self.in_c = [m.out_channels for m in self.stem if isinstance(m, nn.Conv2d) ][-1]

        self.layer1 = self.make_layer(inner_c=64, out_c=256, blocks=3, stride=1)
        self.layer2 = self.make_layer(inner_c=128, out_c=512, blocks=4, stride=2)
        self.layer3 = self.make_layer(inner_c=256, out_c=1024, blocks=6, stride=2)
        self.layer4 = self.make_layer(inner_c=512, out_c=2048, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def make_layer(self, inner_c, out_c, blocks, stride):
        layer = []

        layer += [RNBlock(self.in_c, inner_c, out_c, stride)]

        self.in_c = out_c

        for _ in range(blocks - 1):
            layer += RNBlock(self.in_c, inner_c, out_c, 1)

        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, dim=1)
        out = self.classifier(x)

        return out

class TextEncoder(nn.Module):
    def __init__(self,):
        super().__init__()


class CLIP(nn.Module):
    def __init__(self,):
        super().__init__()