import torch
import model
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_size = 320000
batch_size = 3
src_len = 20
trg_len = 15

src = torch.randint(1, vocab_size, (batch_size, src_len))
trg = torch.randint(1, vocab_size, (batch_size, trg_len))

model = model.Transformer(vocab_size=vocab_size, max_len=50, pad_inx=0, n_layers=6, nhead=8, d_model=512, d_ff=2048, drop_p=0.1)
summary(model, input_data=(src, trg), device='cpu')
print(f'FLOPs: {FlopCountAnalysis(model, (src, trg)).total()}')


src = src.to(device)
trg = trg.to(device)
model = model.to(device)

out = model(src, trg)
print(out.shape)