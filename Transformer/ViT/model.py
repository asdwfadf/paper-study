import torch
import torch.nn as nn
from einops import rearrange


class MHA(nn.Module):
    def __init__(self, hidden_size, heads):
        super().__init__()

        self.heads = heads

        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size, hidden_size)

        self.scale = (hidden_size // heads) ** 0.5

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        Q = rearrange(Q, 'B Seq (C H) -> B H Seq C', H=self.heads)
        K = rearrange(K, 'B Seq (C H) -> B H Seq C', H=self.heads)
        V = rearrange(V, 'B Seq (C H) -> B H Seq C', H=self.heads)

        atten_score = Q @ K.transpose(-2, -1) / self.scale

        atten_weight = torch.softmax(atten_score, dim=-1)
        atten_weight = self.dropout(atten_weight)

        out = atten_weight @ V

        out = rearrange(out, 'B H Seq C -> B Seq (H C)')

        out = self.dropout(self.fc(out))

        return out

class Encoder(nn.Module):
    def __init__(self, hidden_size, mlp_size, heads):
        super().__init__()

        self.mha_LN = nn.LayerNorm(hidden_size)
        self.mha = MHA(hidden_size, heads)

        self.mlp_LN = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_size, hidden_size),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        residual = self.mha_LN(x)
        residual = self.mha(residual)
        x = x + residual

        residual = self.mlp_LN(x)
        residual = self.mlp(residual)
        x = x + residual

        return x


class ViTBase(nn.Module):
    def __init__(self, img_size=224, k=16, layers=12, hidden_size=768, mlp_size=3072, heads=12, pretraining=False, num_classes=1000):
        super().__init__()

        self.layers = layers
        self.patch_size = img_size // k

        self.embedding = nn.Conv2d(3, hidden_size, k, stride=k,) # batch, 768, 14, 14
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.pos = nn.Parameter(torch.randn(1, self.patch_size ** 2 + 1, hidden_size))
        self.embedding_dropout = nn.Dropout(0.1)

        self.encoder = nn.ModuleList([Encoder(hidden_size, mlp_size, heads) for _ in range(layers)])

        if pretraining:
            self.mlp_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, num_classes)
            )
        else:
            self.mlp_head = nn.Linear(hidden_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        x = self.embedding(x)

        x = rearrange(x, 'B C H W -> B (H W) C')

        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos
        x = self.embedding_dropout(x)

        for layer in self.encoder:
            x = layer(x)

        cls = x[:, 0, :]

        out = self.mlp_head(cls)

        return out