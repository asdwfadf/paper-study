import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.reduction_ratio = 16

        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_c, in_c//self.reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_c//self.reduction_ratio, in_c),
            nn.Sigmoid(),
        )

    def forward(self, x):
        se = self.squeeze(x)
        se = se.reshape(x.shape[0], x.shape[1])
        se = self.excitation(se)
        se = se.unsqueeze(2).unsqueeze(3)

        scale = x * se

        return scale


class SEBottleneck(nn.Module):
    def __init__(self, in_c, inner_c, expansion=4, stride=1):
        super().__init__()

        self.expansion = expansion

        self.residual = nn.Sequential(
            nn.Conv2d(in_c, inner_c, 1, stride=1, bias=False,),
            nn.BatchNorm2d(inner_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_c, inner_c, 3, stride=stride, padding=1, bias=False,),
            nn.BatchNorm2d(inner_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_c, inner_c*self.expansion, 1, stride=1, bias=False,),
            nn.BatchNorm2d(inner_c*self.expansion),
        )

        self.se_block = SEBlock(inner_c*self.expansion)

        self.shortcut = nn.Identity()
        if stride != 1 or in_c != inner_c*self.expansion:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_c, inner_c*self.expansion, 1, stride=stride, bias=False,),
                        nn.BatchNorm2d(inner_c*self.expansion),
                    )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)

        residual = self.residual(x)

        scale = self.se_block(residual)

        out = self.relu(shortcut + scale)

        return out


class SEResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_c = 64
        self.expansion = 4

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.in_c, 7, stride=2, padding=3, bias=False,),
            nn.BatchNorm2d(self.in_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.stage1 = self.make_stage(64, 3, 1)
        self.stage2 = self.make_stage(128, 4, 2)
        self.stage3 = self.make_stage(256, 6, 2)
        self.stage4 = self.make_stage(512, 3, 2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

    def make_stage(self, inner_c, layer_num, stride,):
        layer = []

        layer.append(SEBottleneck(self.in_c, inner_c, stride=stride))
        self.in_c = inner_c*self.expansion

        for _ in range(layer_num - 1):
            layer.append(SEBottleneck(self.in_c, inner_c, stride=1))

        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
