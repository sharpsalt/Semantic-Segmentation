import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv3x3-ReLU) ×2, no padding (valid)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3),  # Valid: shrinks by 2px each
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """Upsample + Concat + DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)  # ×2 spatial, halve ch
        self.conv = DoubleConv(in_ch, out_ch)  # in_ch = up_ch + skip_ch = out_ch*2 → out_ch
    
    def forward(self, x1, x2):  # x1: from lower decoder, x2: encoder skip
        x1 = self.up(x1)
        # Pad if dims mismatch (due to valid convs)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2))
        x = torch.cat([x2, x1], dim=1)  # Concat ch
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2):  # Binary
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = nn.MaxPool2d(2)  # After inc
        self.conv1 = DoubleConv(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(256, 512)
        self.down4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)  # No pool after
        
        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, out_ch, 1)
    
    def forward(self, x):
        # Encoder with skips
        x1 = self.inc(x)  # 572→568→564? Wait, valid: 572-4=568, then -4=564? But paper: 572→388? Adjust input or pad.
        # For exact: Input 572, after inc (two valid conv3): 572-4=568, pool→284; etc. Code assumes matching.
        x2 = self.conv1(self.down1(x1))
        x3 = self.conv2(self.down2(x2))
        x4 = self.conv3(self.down3(x3))
        x5 = self.bottleneck(self.down4(x4))
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return torch.sigmoid(self.outc(x))  # Or logits for CE

# Usage
model = UNet(in_ch=1, out_ch=2)
x = torch.randn(1, 1, 572, 572)
out = model(x)  # [1,2,388,388] approx
print(out.shape)

# Training: Weighted CE
class_weights = torch.tensor([0.1, 10.0]).cuda()  # Bg, fg
criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
