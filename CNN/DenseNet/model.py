import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, inner_c, k=32,):
        super().__init__()

        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(inner_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_c, k*4, 1, stride=1, bias=False,),
            nn.BatchNorm2d(k*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(k*4, k, 3, stride=1, padding=1, bias=False,),
        )

    def forward(self, x):
        return self.dense_layer(x)
    

class TransitionLayer(nn.Module):
    def __init__(self, inner_c, compression=0.5,):
        super().__init__()

        out_c = int(inner_c*compression)

        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(inner_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_c, out_c, 1, stride=1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.transition_layer(x)


class DenseNet121(nn.Module):
    def __init__(self, k=32, compression=0.5, num_classes=1000):
        super().__init__()

        self.k = k
        self.compression = compression
        self.inner_c = k*2 # 64
        self.transition = True

        self.stem = nn.Sequential(
                    nn.BatchNorm2d(3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(3, self.k*2, 7, stride=2, padding=3, bias=False,),
                    nn.MaxPool2d(3, stride=2, padding=1),
                )
                
        self.stage1 = self.make_stage(6,)
        self.inner_c = int(self.inner_c * self.compression)
        
        self.stage2 = self.make_stage(12,)
        self.inner_c = int(self.inner_c * self.compression)
        
        self.stage3 = self.make_stage(24,)
        self.inner_c = int(self.inner_c * self.compression)
        self.transition = False

        self.stage4 = self.make_stage(16,)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.inner_c, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu",)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def make_stage(self, layer_num,):
        layer = []
        
        for _ in range(layer_num):
            layer.append(DenseLayer(self.inner_c, self.k,))
            self.inner_c = self.inner_c + self.k

        if self.transition:
            layer.append(TransitionLayer(self.inner_c, self.compression))

        return nn.Sequential(*layer)
    
    def forward_stage(self, concat_x, stage):

        for layer in stage:
            x = layer(concat_x)

            if not isinstance(layer, TransitionLayer):
                concat_x = torch.cat([concat_x, x], dim=1)
            else:
                concat_x = x

        return concat_x

    def forward(self, x):
        concat_x = self.stem(x)

        concat_x = self.forward_stage(concat_x, self.stage1)
        concat_x = self.forward_stage(concat_x, self.stage2)
        concat_x = self.forward_stage(concat_x, self.stage3)
        concat_x = self.forward_stage(concat_x, self.stage4)

        x = self.avgpool(concat_x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
        