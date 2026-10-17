import torch
from torch import nn
from torch.nn import functional as F
import timm

VARIANTS = ('densenet', 'xception', 'fusion', 'convlstm', 'full')


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(channels, max(channels // 16, 1), 1),
                                 nn.ReLU(), nn.Conv2d(max(channels // 16, 1), channels, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        x = x * torch.sigmoid(self.mlp(x.mean((2, 3), keepdim=True)) +
                              self.mlp(x.amax((2, 3), keepdim=True)))
        return x * torch.sigmoid(self.spatial(torch.cat([
            x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))


class ConvLSTM(nn.Module):
    """Two steps represent backbone features of ONE image, not time or slices."""
    def __init__(self, channels):
        super().__init__()
        self.gates = nn.Conv2d(2 * channels, 4 * channels, 3, padding=1)

    def forward(self, sequence):
        h = torch.zeros_like(sequence[0])
        c = torch.zeros_like(h)
        for x in sequence:
            i, f, o, g = self.gates(torch.cat([x, h], 1)).chunk(4, 1)
            c = f.sigmoid() * c + i.sigmoid() * g.tanh()
            h = o.sigmoid() * c.tanh()
        return h


class NeuroFusion(nn.Module):
    def __init__(self, variant='full', pretrained=True, channels=128, dropout=0.4):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        names = ['densenet201', 'legacy_xception']
        if variant == 'densenet':
            names = names[:1]
        elif variant == 'xception':
            names = names[1:]
        self.backbones = nn.ModuleList()
        self.projections = nn.ModuleList()
        for index, name in enumerate(names):
            overlay = {'hf_hub_id': '', 'url': 'https://download.pytorch.org/models/densenet201-c1103571.pth'} if name == 'densenet201' else None
            backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, pretrained_cfg_overlay=overlay)
            self.backbones.append(backbone)
            self.projections.append(nn.Sequential(
                nn.Conv2d(backbone.num_features, channels, 1, bias=False),
                nn.GroupNorm(8, channels), nn.ReLU()))
            cfg = backbone.pretrained_cfg
            self.register_buffer(f'mean_{index}', torch.tensor(cfg['mean']).view(1, 3, 1, 1))
            self.register_buffer(f'std_{index}', torch.tensor(cfg['std']).view(1, 3, 1, 1))
        self.fusion = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.ReLU()) if variant == 'fusion' else None
        self.recurrent = ConvLSTM(channels) if variant in ('convlstm', 'full') else None
        self.attention = CBAM(channels) if variant == 'full' else nn.Identity()
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(channels, 1024), nn.ReLU(), nn.Linear(1024, 3))
        self.frozen = False

    def freeze_backbones(self, frozen):
        self.frozen = frozen
        for backbone in self.backbones:
            backbone.requires_grad_(not frozen)
            if frozen:
                backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbones.eval()
        return self

    def forward_features(self, x):
        features = []
        for i, (backbone, project) in enumerate(zip(self.backbones, self.projections)):
            normalized = (x - getattr(self, f'mean_{i}')) / getattr(self, f'std_{i}')
            features.append(project(F.adaptive_avg_pool2d(backbone.forward_features(normalized), (7, 7))))
        if len(features) == 1:
            merged = features[0]
        elif self.recurrent is not None:
            merged = self.recurrent(features)
        else:
            merged = self.fusion(torch.cat(features, 1))
        return self.attention(merged)

    def forward(self, x):
        return self.head(self.forward_features(x))
