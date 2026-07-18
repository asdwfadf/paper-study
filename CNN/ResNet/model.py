import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False,),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False,),
            nn.BatchNorm2d(out_c),
        )
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False,),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.residual(x)

        out += identity

        out = self.relu(out)

        return out


class ResNet34(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()

        self.in_c = 64

        # Conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False,),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,)

        self.layer1 = self._make_layer(out_c=64, blocks=3, stride=1,)
        self.layer2 = self._make_layer(out_c=128, blocks=4, stride=2,)
        self.layer3 = self._make_layer(out_c=256, blocks=6, stride=2,)
        self.layer4 = self._make_layer(out_c=512, blocks=3, stride=2,)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu",)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self, out_c, blocks, stride):

        layers = []

        layers.append(BasicBlock(self.in_c, out_c, stride,))

        self.in_c = out_c

        for _ in range(blocks - 1):
            layers.append(BasicBlock(self.in_c, out_c,))

        return nn.Sequential(*layers)
    

    def forward(self, x):

        x = self.conv1(x)
        
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x