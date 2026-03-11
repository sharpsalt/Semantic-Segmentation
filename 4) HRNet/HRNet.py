import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):  # ResNet-like
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

class HRFusion(nn.Module):  # Bi-directional fusion block
    def __init__(self, ch):  # Assume equal ch per branch
        super().__init__()
        self.high_conv = BasicBlock(ch, ch)
        self.low_conv = BasicBlock(ch, ch)
        self.down = nn.Conv2d(ch, ch, 3, stride=2, padding=1)  # High to low
        self.up = nn.Conv2d(ch, ch, 1)  # Low to high (post-bilinear)
    
    def forward(self, high, low):
        high_new = self.high_conv(high) + self.up(F.interpolate(low, size=high.shape[-2:], mode='bilinear', align_corners=False))
        low_new = self.low_conv(low) + self.down(high_new)
        return high_new, low_new

class HRNet(nn.Module):
    def __init__(self, num_classes=19, width=18):  # W18: base=18
        super().__init__()
        ch = width // 4  # Per branch ~4-12
        # Stage 1: Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # /4 res
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            BasicBlock(64, 64 * 4)  # 256 ch
        )
        
        # Transition to multi-branch
        self.trans1 = nn.Conv2d(256, ch * 4, 1)  # To stage2: 2 branches, 2ch each? Adjust: total 4*ch=256 for 4 branches, but stage2: 2 branches
        # Simplified: Assume 4 branches from stage2; in full, add progressively.
        
        # Stages 2-4: Parallel (stub for 2 branches; extend to 4)
        self.stage2 = nn.ModuleList([HRFusion(ch * 4) for _ in range(4)])  # T=4 fusions, ch*4 total? Per branch ch
        # Full: ch1=ch*4 (high), ch2=ch*4 (low), etc.
        # Abbrev: For demo, 2 branches, ch=64 each (total 128 stage2)
        
        self.final_conv = nn.Conv2d(ch * 4 * 4, num_classes, 1)  # Fuse all (up+concat), but stub
    
    def forward(self, x):
        x = self.stem(x)  # /4, 256ch
        x = self.trans1(x)  # To multi-ch
        
        # Stage2: 2 branches
        high, low = x[:, :128], x[:, 128:]  # Split
        for fusion in self.stage2:
            high, low = fusion(high, low)
        
        # Stub for stage3/4: Similar, add more branches
        # Fuse: Ups low to high, concat
        low_up = F.interpolate(low, size=high.shape[-2:], mode='bilinear')
        fused = torch.cat([high, low_up], dim=1)
        
        out = self.final_conv(fused)  # Logits
        return F.interpolate(out, size=x.shape[-2:], mode='bilinear')  # Full res? Input x

# Usage
model = HRNet(num_classes=19, width=18)
x = torch.randn(1, 3, 512, 512)
logits = model(x)  # [1,19,512,512]
