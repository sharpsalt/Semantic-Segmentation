import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ASPP(nn.Module):
    def __init__(self, in_ch=2048, out_ch=256, rates=[1,6,12,18], dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(  # r=1: 1x1
                nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(),
            ),
            nn.Sequential(  # r=6,12,18: 3x3 atrous
                nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(),
            ) for rate in rates[1:]
        ])
        # Global avg pool branch
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(),
        )
        self.concat_conv = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 1), out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        h, w = x.shape[-2:]
        branches = [b(x) for b in self.branches]
        global_feat = self.global_branch(x)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)
        branches.append(global_feat)
        return self.concat_conv(torch.cat(branches, dim=1))

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=21, backbone='resnet101'):
        super().__init__()
        # Backbone: ResNet, atrous in later blocks (stub: use pretrained)
        resnet = models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Up to layer4 (/32? Adjust for OS=16)
        # For OS=16: Modify layer4 stride=1, dilate (code abbrev; see official impl)
        
        # Low-level features: From layer3 (/8? For concat, use layer3 256ch)
        self.low_level = nn.Conv2d(256, 48, 1)  # Compress for decoder
        
        self.aspp = ASPP(2048, 256)  # On layer4 2048ch
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x):
        # Encoder
        low_feat = self.backbone[0:5](x)  # Stub: layer1-3, get /8 feat (actual: hooks for layer3)
        # Assume low_feat = layer3 (256ch, /8)
        high_feat = self.backbone(low_feat)  # Full to layer4 /16, 2048ch
        
        aspp_out = self.aspp(high_feat)  # /16, 256ch
        aspp_out = F.interpolate(aspp_out, size=low_feat.shape[-2:], mode='bilinear', align_corners=False)  # To /8
        
        low_out = self.low_level(low_feat)  # /8, 48ch
        dec_in = torch.cat([aspp_out, low_out], dim=1)  # /8, 304ch
        
        dec_out = self.decoder(dec_in)  # /8, C ch
        out = F.interpolate(dec_out, size=x.shape[-2:], mode='bilinear', align_corners=False)  # ×8 to full
        
        return out  # Logits

# Usage
model = DeepLabV3Plus(num_classes=21)
x = torch.randn(1, 3, 513, 513)
logits = model(x)  # [1,21,513,513]
probs = F.softmax(logits, dim=1)

# Training: CE + aux on ASPP
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
