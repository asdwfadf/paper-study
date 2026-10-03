import torch
import torch.nn as nn
from model import CLIP
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1

out_dim=1024
vocab_size=49408
context_length=77
d_model=512
num_layers=12
num_heads=8

model = CLIP(out_dim=out_dim, vocab_size=vocab_size, context_length=context_length, d_model=d_model, num_layers=num_layers, num_heads=num_heads).to(device)
criterion = nn.CrossEntropyLoss()

image = torch.randn(batch_size, 3, 224, 224).to(device)
text = torch.randint(0, vocab_size, (batch_size, context_length)).to(device)

image_text_similarity, text_image_similarity = model(image, text)
labels = torch.arange(batch_size).to(device)

image_loss = criterion(image_text_similarity, labels)
text_loss = criterion(text_image_similarity, labels)
loss = (image_loss + text_loss) / 2

print(f'image_text shape : {image_text_similarity.shape}')
print(f'image loss : {image_loss}')
print(f'text_image shape : {text_image_similarity.shape}')
print(f'text loss : {text_loss}')
print(f'loss : {loss}')

summary(model, input_data=(image, text))
print(f'FLOPs: {FlopCountAnalysis(model, (image, text)).total()}')