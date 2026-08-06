import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    def __init__(self, in_c, r=4):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_c, in_c // r),
            nn.ReLU(inplace=True),
            nn.Linear(in_c // r, in_c),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        se = self.squeeze(x)
        se = se.reshape(x.shape[0], x.shape[1])
        se = self.excitation(se)
        se = se.unsqueeze(2).unsqueeze(3)

        scale = x * se

        return scale


class DepSESep(nn.Module):
    def __init__(self, in_c, out_c, filter_size, use_SE, NL, s, p):
        super().__init__()

        self.use_SE = use_SE

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_c, in_c, filter_size, stride=s, padding=p, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c, momentum=0.99),
            nn.Hardswish(inplace=True) if NL == 'HS' else nn.ReLU(inplace=True)
        )

        if self.use_SE is not None:
            self.SE = SqueezeExcitation(in_c, 4)

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_c, momentum=0.99),
        )

    def forward(self, x):
        x = self.depthwise(x)

        if self.use_SE is not None:
            x = self.SE(x)

        x = self.pointwise(x)

        return x
        

class InvertedBlock(nn.Module):
    def __init__(self, in_c, filter_size, exp_size, out_c, use_SE, NL, s):
        super().__init__()

        self.use_skip_connection = (s == 1 and in_c == out_c)

        if filter_size == 3:
            p = 1
        elif filter_size == 5:
            p = 2

        layer = []
        if in_c != exp_size:
            layer += [nn.Sequential(
                nn.Conv2d(in_c, exp_size, 1, stride=1, bias=False),
                nn.BatchNorm2d(exp_size, momentum=0.99),
                nn.Hardswish(inplace=True) if NL == 'HS' else nn.ReLU(inplace=True)
            )]

        layer += [nn.Sequential(
            DepSESep(exp_size, out_c, filter_size, use_SE, NL, s, p)
        )]

        self.residual = nn.Sequential(*layer)

    def forward(self, x):
        if self.use_skip_connection:
            x = x + self.residual(x)
        else:
            x = self.residual(x)

        return x

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        cfgs = [ # filter_size, exp_size, out_c, use_SE, NL, s
            [3, 16, 16, True, 'RE', 2],
            [3, 72, 24, None, 'RE', 2],
            [3, 88, 24, None, 'RE', 1],
            [5, 96, 40, True, 'HS', 2],
            [5, 240, 40, True, 'HS', 1],
            [5, 240, 40, True, 'HS', 1],
            [5, 120, 48, True, 'HS', 1],
            [5, 144, 48, True, 'HS', 1],
            [5, 288, 96, True, 'HS', 2],
            [5, 576, 96, True, 'HS', 1],
            [5, 576, 96, True, 'HS', 1],
        ]

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16, momentum=0.99),
            nn.Hardswish(inplace=True)
        )

        in_c = 16
        layers = []
        for filter_size, exp_size, out_c, use_SE, NL, s in cfgs:
            layers += [InvertedBlock(in_c, filter_size, exp_size, out_c, use_SE, NL, s)]
            in_c = out_c
        self.layers = nn.Sequential(*layers)

        self.conv_tail = nn.Sequential(
            nn.Conv2d(in_c, 576, 1, stride=1, bias=False),
            nn.BatchNorm2d(576, momentum=0.99),
            nn.Hardswish(inplace=True),
            # SqueezeExcitation(576, 4),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.conv_tail(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x