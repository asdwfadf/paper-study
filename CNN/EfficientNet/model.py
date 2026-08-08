import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.ops import StochasticDepth
import math

class SqueezeExcitation(nn.Module):
    def __init__(self, in_c, squeeze_c, r=4):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_c, squeeze_c),
            nn.SiLU(inplace=True),
            nn.Linear(squeeze_c, in_c),
            nn.Sigmoid()
        )

    def forward(self, x):
        se = self.squeeze(x)
        se = se.reshape(x.shape[0], x.shape[1])
        se = self.excitation(se)
        se = se.unsqueeze(2).unsqueeze(3)

        scale = x * se

        return scale

class DepSESep(nn.Module):
    def __init__(self, in_c, squeeze_c, out_c, k, stride):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_c, in_c, k, stride=stride, padding=k//2, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c, momentum=0.01),
            nn.SiLU(inplace=True)
        )

        self.SE = SqueezeExcitation(in_c, squeeze_c, 4)

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_c, momentum=0.01),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.SE(x)
        x = self.pointwise(x)

        return x

class MBConv(nn.Module):
    def __init__(self, in_c, exp_c, out_c, k, stride, sd_prob=0.2):
        super().__init__()

        self.use_skip_connection = (stride == 1 and in_c == out_c)
        self.stochastic_depth = StochasticDepth(sd_prob, "row")

        layers = []
        if in_c != exp_c:
            layers += [nn.Sequential(
                nn.Conv2d(in_c, exp_c, 1, stride=1, bias=False),
                nn.BatchNorm2d(exp_c, momentum=0.01),
                nn.SiLU(inplace=True)
            )]

        squeeze_c = in_c // 4

        layers += [nn.Sequential(
            DepSESep(exp_c, squeeze_c, out_c, k, stride,)
        )]

        self.residual = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_skip_connection:
            residual = self.residual(x)
            residual = self.stochastic_depth(residual)
            return x + residual
        else:
            return self.residual(x)

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000, depth_mult=1.0, width_mult=1.0, resize_size=256, crop_size=224):
        super().__init__()

        self.transforms = transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        cfgs = [ # k, t, out_c, L, s
            [3, 1, 16, 1, 1],
            [3, 6, 24, 2, 2],
            [5, 6, 40, 2, 2],
            [3, 6, 80, 3, 2],
            [5, 6, 112, 3, 1],
            [5, 6, 192, 4, 2],
            [3, 6, 320, 1, 1],
        ]

        in_c = self.make_divisible(32 * width_mult)
        last_c = self.make_divisible(1280 * width_mult)

        self.stem = nn.Sequential(
            nn.Conv2d(3, in_c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_c, momentum=0.01),
            nn.SiLU(inplace=True)
        )

        layers = []
        for k, t, out_c, L, s in cfgs:
            L = math.ceil(L * depth_mult)
            for i in range(L):
                stride = s if i == 0 else 1
                exp_c = self.make_divisible(in_c * t)
                out_c = self.make_divisible(out_c * width_mult)
                layers += [MBConv(in_c, exp_c, out_c, k, stride, sd_prob=0.2)]
                in_c = out_c

        self.layers = nn.Sequential(*layers)

        self.conv_tail = nn.Sequential(
            nn.Conv2d(in_c, last_c, 1, stride=1, bias=False,),
            nn.BatchNorm2d(last_c, momentum=0.01),
            nn.SiLU(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_c, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / torch.sqrt(torch.tensor(m.out_features))
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor

        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

        # 원래 값보다 10% 이상 감소하지 않도록 보정
        if new_v < 0.9 * v:
            new_v += divisor

        return new_v

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.conv_tail(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x