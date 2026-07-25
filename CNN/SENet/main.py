import torch
from torchinfo import summary
import model

x = torch.randn(2, 3, 224, 224)

model = model.SEResNet50()
summary(model, input_size=x.shape, device='cpu')

out = model(x)
print(out.shape)