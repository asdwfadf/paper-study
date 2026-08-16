import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# Utility Functions
# -------------------------------------------------

def window_partition(x, window_size):
    """
    x: [B, H, W, C]
    return: [num_windows * B, window_size, window_size, C]
    """
    B, H, W, C = x.shape

    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C
    )

    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, C)
    )

    return windows


def window_reverse(windows, window_size, H, W):
    """
    windows: [num_windows * B, window_size, window_size, C]
    return: [B, H, W, C]
    """
    num_windows = (H // window_size) * (W // window_size)
    B = windows.shape[0] // num_windows
    C = windows.shape[-1]

    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        C
    )

    x = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(B, H, W, C)
    )

    return x


# -------------------------------------------------
# MLP
# -------------------------------------------------

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)

        return x


# -------------------------------------------------
# Window-based Multi-Head Self-Attention
# -------------------------------------------------

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout=0.0):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        relative_position_size = (2 * window_size - 1) ** 2

        self.relative_position_bias_table = nn.Parameter(torch.zeros(relative_position_size, num_heads))

        # Relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)

        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))

        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (coords_flatten[:, :, None] - coords_flatten[:, None, :])

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1

        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        x: [B_windows, N, C]
        mask: [num_windows, N, N] or None
        """
        B_windows, N, C = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(B_windows, N, 3, self.num_heads, C // self.num_heads)

        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = q @ k.transpose(-2, -1)

        relative_position_bias = (
            self.relative_position_bias_table[self.relative_position_index.view(-1)]
            .view(N, N, self.num_heads)
            .permute(2, 0, 1)
            .contiguous()
        )

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]

            attn = attn.view(B_windows // num_windows, num_windows, self.num_heads, N, N)

            attn = attn + mask.unsqueeze(1).unsqueeze(0)

            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v

        x = x.transpose(1, 2).reshape(B_windows, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# -------------------------------------------------
# Swin Transformer Block
# -------------------------------------------------

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        H, W = input_resolution

        # Feature map보다 window가 큰 경우에 대한 처리
        if min(H, W) <= window_size:
            self.window_size = min(H, W)
            self.shift_size = 0

        self.norm1 = nn.LayerNorm(dim)

        self.attn = WindowAttention(dim=dim, window_size=self.window_size, num_heads=num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)

        self.mlp = MLP(dim=dim, hidden_dim=hidden_dim, dropout=dropout)

        if self.shift_size > 0:
            attn_mask = self.create_attention_mask(H, W)
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def create_attention_mask(self, H, W):
        """
        Shifted Window에서 서로 다른 영역의 패치가
        잘못 연결되지 않도록 attention mask를 생성합니다.
        """
        img_mask = torch.zeros((1, H, W, 1))

        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )

        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )

        count = 0

        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = count
                count += 1

        mask_windows = window_partition(img_mask, self.window_size)

        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        attn_mask = (mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2))

        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf"))

        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

        return attn_mask

    def forward(self, x):
        """
        x: [B, H*W, C]
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W

        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic Shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window Partition
        x_windows = window_partition(shifted_x, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window Attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Window Reverse
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse Cyclic Shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # Residual Connection
        x = shortcut + x

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


# -------------------------------------------------
# Patch Merging
# -------------------------------------------------

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim

        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        """
        x: [B, H*W, C]
        output: [B, H/2*W/2, 2C]
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W
        assert H % 2 == 0 and W % 2 == 0

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)

        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)

        return x


# -------------------------------------------------
# Swin Stage
# -------------------------------------------------

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7, mlp_ratio=4.0, dropout=0.0, downsample=True):
        super().__init__()

        self.blocks = nn.ModuleList()

        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                )
            )

        if downsample:
            self.downsample = PatchMerging(input_resolution=input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


# -------------------------------------------------
# Swin Transformer
# -------------------------------------------------

class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        # Patch Partition + Linear Embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.patch_norm = nn.LayerNorm(embed_dim)

        resolution = img_size // patch_size

        self.layers = nn.ModuleList()

        dim = embed_dim

        for i in range(len(depths)):
            downsample = i < len(depths) - 1

            self.layers.append(
                BasicLayer(
                    dim=dim,
                    input_resolution=(resolution, resolution),
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    downsample=downsample
                )
            )

            if downsample:
                resolution //= 2
                dim *= 2

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x):
        """
        x: [B, 3, H, W]
        """
        x = self.patch_embed(x)

        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H * W, C)

        x = self.patch_norm(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Global Average Pooling
        x = x.mean(dim=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x