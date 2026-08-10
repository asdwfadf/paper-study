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

        self.scale = torch.sqrt(d_model / nhead)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask = None):
        Q = self.Q(q) # 개, 단, 차
        K = self.K(k) # 개, 단, 차
        V = self.V(v) # 개, 단, 차

        Q = rearrange(Q, '개 단 (헤 차) -> 개 헤 단 차', 헤 = self.nhead)
        K = rearrange(K, '개 단 (헤 차) -> 개 헤 단 차', 헤 = self.nhead)
        V = rearrange(V, '개 단 (헤 차) -> 개 헤 단 차', 헤 = self.nhead)

        attention_score = Q * K.transpose(-2, -1) / self.scale # 개, 헤, 단, 단

        if mask is not None:
            attention_score[mask] = -1e10

        attention_weight = torch.softmax(attention_score, -1) # 개, 헤, 단, 단

        attention = attention_weight * V # 개, 헤, 단, 차

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

        layers = []
        for i in range(n_layers):
            layers += [EncoderLayer(nhead, d_model, d_ff, drop_p)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x, pad_mask):
        x = self.layers(x, pad_mask)

        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, pad_inx, n_layers, nhead, d_ff, drop_p, d_model=512, max_len=50,):
        super().__init__()

        self.pad_inx = pad_inx

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)

        self.encoder = Encoder(n_layers, nhead, d_model, d_ff, drop_p)

    def make_pad_mask(self, x, pad_inx):
        pad_mask = (x == pad_inx).unsqueeze(1).unsquezze(2)
        pad_mask = pad_mask.expand(x.shape[0], self.nhead, x.shape[1], x.shape[1])

        return pad_mask

    def make_dec_mask(self, x,):
        future_mask = torch.tril(torch.ones(x.shape[0], self.nhead, x.shape[1], x.shape[1]))

        return future_mask

    def forward(self, src, trg):
        pad_mask = self.make_pad_mask(src, self.pad_inx)
        future_mask = self.make_dec_mask(trg)

        src = self.embedding(src)
        src = self.pos(src)

        enc_out = self.encoder(src, pad_mask)