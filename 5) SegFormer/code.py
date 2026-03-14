import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# DropPath (stochastic depth)
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


# Efficient Multi-Head Self-Attention with Spatial Reduction (used in MiT)
class EfficientAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        sr_ratio: int = 1,
        qkv_bias: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, padding=0)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape

        # Query
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Key & Value (with spatial reduction for efficiency)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Weighted sum
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# Mix-FFN (MLP + 3x3 depthwise conv for spatial mixing)
class MixFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=True
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape

        # First linear
        x = self.fc1(x)

        # Reshape to spatial for depthwise conv
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # B, C, H, W
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).reshape(B, N, -1)  # back to B, N, C

        x = self.act(x)
        x = self.drop(x)

        # Second linear
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Transformer Block used in each stage of MiT
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        sr_ratio: int = 1,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, mlp_ratio, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


# Overlapped Patch Embedding (with overlap for better local context)
class OverlapPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 7,
        stride: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)
        return x, H, W


# Mix Transformer Backbone (MiT) - hierarchical encoder
class MiT(nn.Module):
    def __init__(
        self,
        embed_dims: List[int] = [32, 64, 160, 256],
        num_heads: List[int] = [1, 2, 5, 8],
        mlp_ratios: List[float] = [4.0, 4.0, 4.0, 4.0],
        depths: List[int] = [2, 2, 2, 2],
        sr_ratios: List[int] = [8, 4, 2, 1],
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.num_stages = len(embed_dims)

        self.patch_embeds = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(self.num_stages):
            # Patch embedding
            if i == 0:
                patch_size, stride, in_chans = 7, 4, 3
            else:
                patch_size, stride, in_chans = 3, 2, embed_dims[i - 1]

            self.patch_embeds.append(
                OverlapPatchEmbed(patch_size, stride, in_chans, embed_dims[i])
            )

            # Transformer blocks for this stage
            stage_blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        sr_ratio=sr_ratios[i],
                        drop_path=dpr[cur + j],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage_blocks)

            self.norms.append(nn.LayerNorm(embed_dims[i]))
            cur += depths[i]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        B = x.shape[0]

        for i in range(self.num_stages):
            # Patch embedding
            x, H, W = self.patch_embeds[i](x)

            # Transformer blocks
            for blk in self.stages[i]:
                x = blk(x, H, W)

            # Norm + reshape back to spatial
            x = self.norms[i](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            features.append(x)

        return features


# Lightweight SegFormer Decoder (MLP-style fusion)
class SegFormerHead(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        embed_dim: int = 256,
        num_classes: int = 19,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # 1x1 projections (implemented as Linear for exact paper match)
        self.linear_c = nn.ModuleList(
            [nn.Linear(ch, embed_dim) for ch in in_channels]
        )

        # Fusion + prediction
        self.linear_fuse = nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        B = features[0].shape[0]
        target_h, target_w = features[0].shape[2:]  # highest resolution (1/4)

        outs = []
        for i, feat in enumerate(features):
            # Linear projection (channel reduction)
            feat = feat.permute(0, 2, 3, 1)  # B, H, W, C
            feat = self.linear_c[i](feat)     # B, H, W, embed_dim
            feat = feat.permute(0, 3, 1, 2)   # B, embed_dim, H, W

            # Upsample to 1/4 resolution (stage 0 size)
            if i != 0:
                feat = F.interpolate(
                    feat, size=(target_h, target_w), mode="bilinear", align_corners=False
                )

            outs.append(feat)

        # Concatenate along channel dimension
        fused = torch.cat(outs, dim=1)  # B, 4*embed_dim, H/4, W/4

        # Fusion + prediction
        fused = self.linear_fuse(fused)
        logits = self.linear_pred(fused)
        return logits


# Full SegFormer model (configurable for B0–B5)
class SegFormer(nn.Module):
    def __init__(
        self,
        variant: str = "B0",
        num_classes: int = 19,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        # Official configs from the paper (you can add B1–B5 easily)
        configs = {
            "B0": {
                "embed_dims": [32, 64, 160, 256],
                "num_heads": [1, 2, 5, 8],
                "mlp_ratios": [4.0, 4.0, 4.0, 4.0],
                "depths": [2, 2, 2, 2],
                "sr_ratios": [8, 4, 2, 1],
                "decoder_embed_dim": 256,
            },
            # Add others if needed:
            # "B1": {"embed_dims": [64, 128, 320, 512], "num_heads": [1, 2, 5, 8], ...}
        }

        if variant not in configs:
            raise ValueError(f"Variant {variant} not supported. Use B0 (or extend configs).")

        cfg = configs[variant]

        self.backbone = MiT(
            embed_dims=cfg["embed_dims"],
            num_heads=cfg["num_heads"],
            mlp_ratios=cfg["mlp_ratios"],
            depths=cfg["depths"],
            sr_ratios=cfg["sr_ratios"],
            drop_path_rate=drop_path_rate,
        )

        self.decode_head = SegFormerHead(
            in_channels=cfg["embed_dims"],
            embed_dim=cfg["decoder_embed_dim"],
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        features = self.backbone(x)

        # Decoder (outputs at 1/4 resolution)
        logits = self.decode_head(features)

        # Upsample to original input resolution
        logits = F.interpolate(
            logits, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        return logits


# ================ Example Usage ================
if __name__ == "__main__":
    model = SegFormer(variant="B0", num_classes=19)  # e.g., Cityscapes has 19 classes
    model.eval()

    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    print("Output shape:", out.shape)  # torch.Size([1, 19, 512, 512])
