import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]

        return x


class MHA(nn.Module):
    def __init__(self, nhead, d_model):
        super().__init__()

        self.nhead = nhead

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)

        self.scale = (d_model / nhead) ** 0.5

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask = None):
        Q = self.Q(q) # 개, 단, 차
        K = self.K(k) # 개, 단, 차
        V = self.V(v) # 개, 단, 차

        Q = rearrange(Q, '개 단 (헤 차) -> 개 헤 단 차', 헤 = self.nhead)
        K = rearrange(K, '개 단 (헤 차) -> 개 헤 단 차', 헤 = self.nhead)
        V = rearrange(V, '개 단 (헤 차) -> 개 헤 단 차', 헤 = self.nhead)

        attention_score = Q @ K.transpose(-2, -1) / self.scale # 개, 헤, 단, 단

        if mask is not None:
            attention_score[mask] = -1e10

        attention_weight = torch.softmax(attention_score, -1) # 개, 헤, 단, 단

        attention = attention_weight @ V # 개, 헤, 단, 차

        x = rearrange(attention, '개 헤 단 차 -> 개 단 (헤 차)') # 개, 단, 차

        x = self.fc(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_p):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_p),
            nn.Linear(d_ff, d_model)
            )

    def forward(self, x):
        x = self.fc(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, nhead, d_model, d_ff, drop_p):
        super().__init__()

        self.nhead = nhead

        self.mha = MHA(nhead, d_model)
        self.mhaLN = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model, d_ff, drop_p)
        self.ffLN = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, pad_mask):
        residual = self.mha(x, x, x, pad_mask)
        residual = self.dropout(residual)
        x = self.mhaLN(x + residual)

        residual = self.ff(x)
        residual = self.dropout(residual)
        x = self.ffLN(x + residual)

        return x


class Encoder(nn.Module):
    def __init__(self, n_layers, nhead, d_model, d_ff, drop_p):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(nhead, d_model, d_ff, drop_p) for _ in range(n_layers)])

    def forward(self, x, pad_mask):
        for layer in self.layers:
            x = layer(x, pad_mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, nhead, d_model, d_ff, drop_p):
        super().__init__()

        self.mha = MHA(nhead, d_model)
        self.mhaLN = nn.LayerNorm(d_model)

        self.cross_mha = MHA(nhead, d_model)
        self.crossLN = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model, d_ff, drop_p)
        self.ffLN = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, enc_out, x, future_mask, cross_mask):
        residual = self.mha(x, x, x, future_mask)
        residual = self.dropout(residual)
        mha_out = self.mhaLN(x + residual)

        residual = self.cross_mha(mha_out, enc_out, enc_out, cross_mask)
        residual = self.dropout(residual)
        cross_out = self.crossLN(mha_out + residual)

        residual = self.ff(cross_out)
        residual = self.dropout(residual)
        dec_out = self.ffLN(cross_out + residual)

        return dec_out

class Decoder(nn.Module):
    def __init__(self, n_layers, nhead, d_model, d_ff, drop_p):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(nhead, d_model, d_ff, drop_p) for _ in range(n_layers)])

    def forward(self, enc_out, x, future_mask, cross_mask):
        for layer in self.layers:
            x = layer(enc_out, x, future_mask, cross_mask)

        return x
    

class Transformer(nn.Module):
    def __init__(self, vocab_size=320000, pad_inx=0, n_layers=6, nhead=8, d_ff=2048, drop_p=0.1, d_model=512, max_len=50,):
        super().__init__()

        self.nhead = nhead
        self.pad_inx = pad_inx
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(drop_p)

        self.encoder = Encoder(n_layers, nhead, d_model, d_ff, drop_p)
        self.decoder = Decoder(n_layers, nhead, d_model, d_ff, drop_p)

        self.fc = nn.Linear(d_model, vocab_size)

        for m in self.modules():
            if hasattr(m,'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)

    def make_enc_mask(self, x):
        pad_mask = (x == self.pad_inx).unsqueeze(1).unsqueeze(2)
        pad_mask = pad_mask.expand(x.shape[0], self.nhead, x.shape[1], x.shape[1])

        return pad_mask

    def make_dec_mask(self, x):
        future_mask = torch.tril(torch.ones(x.shape[0], self.nhead, x.shape[1], x.shape[1], device=x.device))==0

        return future_mask

    def make_cross_mask(self, src, trg):
        cross_mask = (src == self.pad_inx).unsqueeze(1).unsqueeze(2)
        cross_mask = cross_mask.expand(trg.shape[0], self.nhead, trg.shape[1], src.shape[1])

        return cross_mask

    def forward(self, src, trg):
        pad_mask = self.make_enc_mask(src)
        future_mask = self.make_dec_mask(trg)
        cross_mask = self.make_cross_mask(src, trg)
        
        src = self.embedding(src) * (self.d_model ** 0.5)
        src = self.pos(src)
        src = self.dropout(src)
        enc_out = self.encoder(src, pad_mask)

        trg = self.embedding(trg) * (self.d_model ** 0.5)
        trg = self.pos(trg)
        trg = self.dropout(trg)
        dec_out = self.decoder(enc_out, trg, future_mask, cross_mask)

        out = self.fc(dec_out)

        return out