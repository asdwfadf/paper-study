import torch
import model
from torchinfo import summary

x = torch.randn(2, 3, 224, 224)

model = model.MobileNetV2()
print(summary(model, input_size=x.shape, device='cpu'))

out = model(x)

print(out.shape)