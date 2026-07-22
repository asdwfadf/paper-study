import torch
from torchinfo import summary
import model

SEED = 42

x = torch.randn(2, 3, 224, 224)

model = model.DenseNet121()
summary(model, input_size=x.shape, device='cpu')

outputs = model(x)
print(outputs.shape)