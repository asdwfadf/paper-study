import torch
import model
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis

x = torch.randn(1, 3, 224, 224)

model = model.EfficientNetB0()
print(summary(model, input_size=x.shape, device='cpu'))
print(f'FLOPs: {FlopCountAnalysis(model, x).total()}')

out = model(x)
print(out.shape)