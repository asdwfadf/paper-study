import torch
import model
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.randn(1, 3, 224, 224)

model = model.ConvNeXt_T()
summary(model, input_data=x, device='cpu')
print(f'FLOPs: {FlopCountAnalysis(model, x).total()}')

x = x.to(device)
model = model.to(device)

out = model(x)
print(out.shape)