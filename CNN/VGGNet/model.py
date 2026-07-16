import torch.nn as nn

class VGGNet16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGGNet16, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.extract_features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Layer 2
            nn.Conv2d(64, 128, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Layer 3
            nn.Conv2d(128, 256, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Layer 4
            nn.Conv2d(256, 512, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Layer 5
            nn.Conv2d(512, 512, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            # fc1
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # fc2
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # fc3
            nn.Linear(4096, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.extract_features(x)
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)

        return logits
