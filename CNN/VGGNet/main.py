import torch
import torch.optim as optim
import torch.nn as nn
from torchinfo import summary
import model

SEED = 42

torch.manual_seed(SEED)

x = torch.randn(2, 3, 224, 224)

model = model.VGGNet16()
summary(model, input_size=x.shape, device='cpu')

outputs = model(x)
print(outputs.shape)