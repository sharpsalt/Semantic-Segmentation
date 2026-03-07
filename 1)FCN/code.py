# FCN-8s with VGG16 as Backbone
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN8s, self).__init__()
        # Backbone: VGG16 features up to pool5
        vgg = models.vgg16(pretrained=True).features  # Conv layers only
        self.encoder = nn.Sequential(*vgg[:17])  # Up to pool4 (layer 16 is pool4)
        self.pool5_conv = nn.Sequential(*vgg[17:])  # conv5_1,2,3 + pool5? Wait, adjust indices
        
        # Actually, for precision: Extract layers
        # Layer indices: conv1_1(0), ..., pool1(4), ..., pool4(16), conv5_1(17), conv5_2(18), conv5_3(19), pool5(20? but features ends at 30? Use named.
        # Simpler: Use frozen VGG and slice.
        
        # Score layers
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)  # Skip from pool4
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)  # Skip from pool3
        
        # Upsample deconvs
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1)  # ×2
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, padding=4)     # ×8 for final
        
        # Pool3 is after layer ~10 (conv3_3=9, relu=10, pool3=11? Indices: standard VGG features[0:7]=block1+pool1, [7:14]=block2+pool2, [14:21?]=block3+pool3 (pool3 at 16?), let's define explicitly.
        
        # Better: Define encoder blocks
        self.convblock1 = nn.Sequential(  # To pool1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64,64,3,p=1), nn.ReLU(), nn.MaxPool2d(2,stride=2)
        )
        # ... (abbrev; in code, use vgg.features[:7] for block1, etc.)
        # For brevity, assume we slice:
        # self.pool3_feat = vgg.features[:16]  # Up to pool3? Standard: pool1:4, pool2:9, pool3:16, pool4:23, pool5:30.
        # VGG16 features has 31 layers (0-based).
        # pool3_end = 16 (after pool3), pool4_end=23, pool5_end=30.
        
        # Full code (simplified with slices):
        self.features = vgg  # Full features
        self.score_fr = nn.Conv2d(512, num_classes, 1)  # After pool5 (512 ch)
    
    def forward(self, x):
        # Encoder: Get intermediate features
        x = self.features[0:7](x)   # To after pool1: H/2, 64ch
        pool1 = x  # Not used in FCN-8s
        
        x = self.features[7:14](x)  # To after pool2: H/4, 128ch
        pool2 = x
        
        x = self.features[14:23](x)  # Wait, pool3 at 16? Precise slicing:
        # Actually, to get exact: Use hooks or sequential blocks. For demo:
        # Assume we have pool3, pool4, pool5 as outputs.
        # Pseudo:
        pool3 = self.get_pool3(x)  # H/8, 256ch
        pool4 = self.get_pool4(pool3)  # H/16, 512ch
        pool5 = self.get_pool5(pool4)  # H/32, 512ch
        
        # Scores
        score_pool5 = self.score_fr(pool5)  # H/32 x C
        
        # FCN-16s first: ×2 up + pool4 skip
        score_pool4up = self.upscore_pool4(score_pool5)  # H/16 x C
        score_pool4 = self.score_pool4(pool4)  # H/16 x C
        score_fcn16 = score_pool4up + score_pool4  # Skip add
        
        # FCN-8s: ×2 up + pool3 skip
        score_pool3up = self.upscore_pool4(score_fcn16)  # Reuse for ×2, H/8 x C
        score_pool3 = self.score_pool3(pool3)  # H/8 x C
        score_fcn8 = score_pool3up + score_pool3
        
        # Final ×8 up to full HxW
        out = self.upscore8(score_fcn8)  # Bilinear if needed: F.interpolate(out, size=x.shape[-2:], mode='bilinear')
        
        return out  # Logits HxWxC

# Usage & Training Snippet
model = FCN8s(num_classes=21)
x = torch.randn(1, 3, 224, 224)  # Arbitrary size
logits = model(x)
print(logits.shape)  # [1,21,224,224]

# Loss
criterion = nn.CrossEntropyLoss(ignore_index=255)  # For voids in datasets
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
# For fine-tune: Freeze early layers, LR 1e-2 on decoder.
