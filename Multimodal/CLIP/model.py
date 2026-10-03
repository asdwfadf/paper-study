import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

class AttentionPool(nn.Module):
    def __init__(self, out_dim=1024, spacial_dim=7, embed_dim=2048, num_heads=32):
        super().__init__()
        
        # 49 + 1, 2048(해상도+special token, out_c)
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)

        self.Q_proj = nn.Linear(embed_dim, embed_dim)
        self.K_proj = nn.Linear(embed_dim, embed_dim)
        self.V_proj = nn.Linear(embed_dim, embed_dim)
        self.C_proj = nn.Linear(embed_dim, out_dim)

        self.num_heads = num_heads
        self.scale = (embed_dim / num_heads) ** 0.5

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(0, 2, 1) # (B, spacial, out_c)
        x = torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)
        x = x + self.positional_embedding

        Q = self.Q_proj(x[:, :1, :]) # x[:1].shape = (B, 1, out_c), x[0].shape = (B, out_c)
        K = self.K_proj(x)
        V = self.V_proj(x)

        Q = rearrange(Q, 'B spacial (out_c heads) -> B heads spacial out_c', heads = self.num_heads)
        K = rearrange(K, 'B spacial (out_c heads) -> B heads spacial out_c', heads = self.num_heads)
        V = rearrange(V, 'B spacial (out_c heads) -> B heads spacial out_c', heads = self.num_heads)

        attn_score = Q @ K.transpose(-2, -1) / self.scale
        attn_weight = torch.softmax(attn_score, -1)
        attention = attn_weight @ V

        x = rearrange(attention, 'B heads spacial out_c -> B spacial (out_c heads)') # B, 1, out_c

        out = self.C_proj(x)

        return out.squeeze(1) # B out_c

class CRNBlock(nn.Module):
    def __init__(self, in_c, inner_c, out_c, stride):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_c, inner_c, 1, stride=1, bias=False),
            nn.BatchNorm2d(inner_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(inner_c, inner_c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inner_c),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(stride),

            nn.Conv2d(inner_c, out_c, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_c),
        )

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(stride),
                nn.Conv2d(in_c, out_c, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        out = self.relu(shortcut + residual)

        return out


class CLIPResNet50(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(2, stride=2)
        )

        self.in_c = [m.out_channels for m in self.stem if isinstance(m, nn.Conv2d) ][-1]

        self.layer1 = self.make_layer(inner_c=64, out_c=256, blocks=3, stride=1)
        self.layer2 = self.make_layer(inner_c=128, out_c=512, blocks=4, stride=2)
        self.layer3 = self.make_layer(inner_c=256, out_c=1024, blocks=6, stride=2)
        self.layer4 = self.make_layer(inner_c=512, out_c=2048, blocks=3, stride=2)

        self.attnpool = AttentionPool(out_dim=out_dim, spacial_dim=7, embed_dim=2048, num_heads=32)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def make_layer(self, inner_c, out_c, blocks, stride):
        layer = []

        layer += [CRNBlock(self.in_c, inner_c, out_c, stride)]

        self.in_c = out_c

        for _ in range(blocks - 1):
            layer += [CRNBlock(self.in_c, inner_c, out_c, 1)]

        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

class MHA(nn.Module):
    def __init__(self, mask, d_model=512, num_heads=8):
        super().__init__()

        self.scale = (d_model / num_heads) ** 0.5
        self.d_model = d_model
        self.num_heads = num_heads
        self.register_buffer("mask", mask)

        self.Q_proj = nn.Linear(d_model, d_model)
        self.K_proj = nn.Linear(d_model, d_model)
        self.V_proj = nn.Linear(d_model, d_model)
        self.O_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = rearrange(self.Q_proj(x), 'B seq (d_model num_heads) -> B num_heads seq d_model', num_heads = self.num_heads)
        K = rearrange(self.K_proj(x), 'B seq (d_model num_heads) -> B num_heads seq d_model', num_heads = self.num_heads)
        V = rearrange(self.V_proj(x), 'B seq (d_model num_heads) -> B num_heads seq d_model', num_heads = self.num_heads)

        attn_score = ((Q @ K.transpose(-2, -1)) + self.mask) / self.scale
        attn_weight = torch.softmax(attn_score, -1)
        attention = attn_weight @ V

        attention = rearrange(attention, 'B num_heads seq d_model -> B seq (d_model num_heads)')

        return self.O_proj(attention)


class TransformerBlock(nn.Module):
    def __init__(self, mask, d_model=512, num_heads=8):
        super().__init__()

        self.ln_1 = nn.LayerNorm(d_model)
        self.mha = MHA(mask=mask, d_model=d_model, num_heads=num_heads)

        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
            )

    def forward(self, x):
        x = x + self.mha(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x

class Transformer(nn.Module):
    def __init__(self, mask, num_layers=12, d_model=512, num_heads=8):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers += [TransformerBlock(mask, d_model=d_model, num_heads=num_heads)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class CLIP(nn.Module):
    def __init__(self, out_dim=1024, vocab_size=49152, context_length=77, d_model=512, num_layers=12, num_heads=8):
        super().__init__()

        self.visual = CLIPResNet50(out_dim=out_dim)
        self.logits_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, d_model))
        mask = self.make_mask(context_length)
        self.transformer = Transformer(mask, num_layers=num_layers, d_model=d_model, num_heads=num_heads)
        self.ln_final = nn.LayerNorm(d_model)
        self.text_proj = nn.Linear(d_model, out_dim, bias=False)

    def make_mask(self, context_length):
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)

        return mask

    def encode_image(self, image):
        
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = self.ln_final(x)

        # CLIP에서 종료 토큰만 추출해서 투영함, 근데 종료 토큰의 id를 제일 큰 숫자로 해놔서 argmax로 찾아냄
        # mask 때문에 이전 토큰들을 못 보는데 종료 토큰은 맨 마지막에 있으니 모든 토큰을 볼 수 있으니 종료 토큰을 사용함
        x = self.text_proj(x[torch.arange(x.shape[0]), text.argmax(dim=-1)])

        return x

    def forward(self, image, text):
        image_feature = self.encode_image(image)
        text_feature = self.encode_text(text)

        image_norm = image_feature / image_feature.norm(dim=1, keepdim=True)
        text_norm = text_feature / text_feature.norm(dim=1, keepdim=True)

        image_text_similarity = self.logits_scale.exp() * image_norm @ text_norm.t()
        text_image_similarity = image_text_similarity.t()

        return image_text_similarity, text_image_similarity
