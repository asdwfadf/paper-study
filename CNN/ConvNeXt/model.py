import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class CNBlock(nn.Module):
    def __init__(self, in_c, layer_scale, stochastic_depth_prob):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_c, in_c, 7, stride=1, padding=3, groups=in_c, bias=False),
            Permute([0,2,3,1]), # 개 행 열 채
            nn.LayerNorm(in_c),
            Permute([0,3,1,2]), # 개 채 행 열
            nn.Conv2d(in_c, in_c*4, 1, stride=1,),
            nn.GELU(),
            nn.Conv2d(in_c*4, in_c, 1, stride=1,)
        )

        self.layer_scale = nn.Parameter(torch.ones(1, in_c, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, 'row')

    def forward(self, x):
        residual = self.layer_scale * self.residual(x)
        residual = self.stochastic_depth(residual)
        x = x + residual

        return x

class ConvNeXt_T(nn.Module):
    def __init__(self, layer_scale=1e-6, stochastic_depth_prob=0.1, num_classes=1000):
        super().__init__()

        cfgs = [
            [96, 3, True],
            [192, 3, True],
            [384, 9, True],
            [768, 3, False],
        ]

        total_block = sum(x[1] for x in cfgs)
        stochastic_depth_prob = torch.linspace(0, stochastic_depth_prob, total_block)

        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 4, stride=4),
            Permute([0,2,3,1]), # 개 행 열 채
            nn.LayerNorm(96),
            Permute([0,3,1,2]), # 개 채 행 열
        )

        layers = []
        block = 0
        for c, loop, downsampling in cfgs:
            for _ in range(loop):
                layers += [CNBlock(c, layer_scale, stochastic_depth_prob[block])]
                block += 1
            if downsampling:
                inner_c = c * 2
                Permute([0,2,3,1]), # 개 행 열 채
                nn.LayerNorm(c),
                Permute([0,3,1,2]), # 개 채 행 열
                layers += [nn.Conv2d(c, inner_c, 2, stride=2)]

        self.layers = nn.Sequential(*layers)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x