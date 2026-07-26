import torch
import torch.nn as nn

class DepSepLayer(nn.Module):
    def __init__(self, in_c, inner_c, stride):
        super().__init__()

        self.Depthwise = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
        )

        self.Pointwise = nn.Sequential(
            nn.Conv2d(in_c, inner_c, 1, stride=1, bias=False),
            nn.BatchNorm2d(inner_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.Depthwise(x)
        x = self.Pointwise(x)

        return x

class MobileNetV1(nn.Module):
    def __init__(self, alpha=0.75, num_classes=1000):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, int(32*alpha), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(32*alpha)),
            nn.ReLU(inplace=True),
        )

        self.depsep1 = DepSepLayer(int(32*alpha), int(64*alpha), 1)
        self.depsep2 = nn.Sequential(
                    DepSepLayer(int(64*alpha), int(128*alpha), 2),
                    DepSepLayer(int(128*alpha), int(128*alpha), 1),
                )
        self.depsep3 = nn.Sequential(
                            DepSepLayer(int(128*alpha), int(256*alpha), 2),
                            DepSepLayer(int(256*alpha), int(256*alpha), 1),
                        )
        self.depsep4 = nn.Sequential(
                                    DepSepLayer(int(256*alpha), int(512*alpha), 2),
                                    *[DepSepLayer(int(512*alpha), int(512*alpha), 1) for _ in range(5)],
                                )
        self.depsep5 = nn.Sequential(
                                    DepSepLayer(int(512*alpha), int(1024*alpha), 2),
                                    DepSepLayer(int(1024*alpha), int(1024*alpha), 1),
                                )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(int(1024*alpha), num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.depsep1(x)
        x = self.depsep2(x)
        x = self.depsep3(x)
        x = self.depsep4(x)
        x = self.depsep5(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x