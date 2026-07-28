import torch
import torch.nn as nn

class InvertedBottleneck(nn.Module):
    def __init__(self, in_c, t, c, stride,):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_c, in_c*t, 1, stride=1, bias=False),
            nn.BatchNorm2d(in_c*t),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(in_c*t, in_c*t, 3, stride=stride, padding=1, groups=in_c*t, bias=False),
            nn.BatchNorm2d(in_c*t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_c*t, c, 1, stride=1, bias=False),
            nn.BatchNorm2d(c),
        )

        if stride == 1 and in_c == c:
            self.use_skip_connection = True
        else:
            self.use_skip_connection = False

    def forward(self, x):
        if self.use_skip_connection:
            return self.residual(x) + x
        else:
            return self.residual(x)



class MobileNetV2(nn.Module):
    def __init__(self, alpha=1.0, num_clasees=1000):
        super().__init__()

        config = [# t, c, n, s
            [1, int(alpha*16), 1, 1],
            [6, int(alpha*24), 2, 2],
            [6, int(alpha*32), 3, 2],
            [6, int(alpha*64), 4, 2],
            [6, int(alpha*96), 3, 1],
            [6, int(alpha*160), 3, 2],
            [6, int(alpha*320), 1, 1],
        ]
        self.stem = nn.Sequential(
            nn.Conv2d(3, int(alpha*32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(alpha*32)),
            nn.ReLU6(inplace=True)
        )

        self.in_c = int(alpha*32)

        self.features = nn.Sequential()

        for t, c, n, s in config:
            self.features.append(self.make_layer(t, c, n, s))

        self.conv_tail = nn.Sequential(
            nn.Conv2d(config[6][1], 1280, 1, stride=1, bias=False,),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_clasees)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def make_layer(self, t, c, n, s):
        layer = []

        for i in range(n):
            if i == 0:
                layer.append(InvertedBottleneck(self.in_c, t, c, s))
                self.in_c = c
            else:
                layer.append(InvertedBottleneck(self.in_c, t, c, 1))

        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.stem(x)

        x = self.features(x)

        x = self.conv_tail(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
