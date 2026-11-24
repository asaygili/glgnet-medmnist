import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def resnet(depth, in_ch, n_cls, size):
    net = getattr(torchvision.models, f'resnet{depth}')(num_classes=n_cls)
    if size == 28:  # MedMNIST v2 small-input variant: 3x3 stride-1 stem, no max-pool
        net.conv1 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False)
        net.maxpool = nn.Identity()
    else:
        net.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    return net


# ---------------------------------------------------------------------------
# Proposed: GLGNet — Gated Local-Global Network
# Each block mixes a multi-scale depthwise (local) branch and a pooled-key
# attention (global) branch through an input-conditioned channel gate.
# ---------------------------------------------------------------------------

class LocalBranch(nn.Module):
    """Multi-scale depthwise conv: channels split across kernels 3/5/7."""
    def __init__(self, c, kernels=(3, 5, 7)):
        super().__init__()
        self.splits = [c // len(kernels)] * len(kernels)
        self.splits[0] += c - sum(self.splits)
        self.convs = nn.ModuleList(nn.Conv2d(s, s, k, 1, k // 2, groups=s) for s, k in zip(self.splits, kernels))
        self.pw = nn.Conv2d(c, c, 1)

    def forward(self, x):
        xs = torch.split(x, self.splits, 1)
        return self.pw(torch.cat([cv(t) for cv, t in zip(self.convs, xs)], 1))


class GlobalBranch(nn.Module):
    """Attention whose keys/values come from an adaptively pooled p x p map: O(HW * p^2)."""
    def __init__(self, c, heads, pool=7):
        super().__init__()
        self.h, self.pool = heads, pool
        self.q = nn.Conv2d(c, c, 1)
        self.kv = nn.Conv2d(c, 2 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).flatten(2).view(B, self.h, C // self.h, H * W).transpose(-1, -2)
        kv = self.kv(F.adaptive_avg_pool2d(x, min(self.pool, H))).flatten(2)
        k, v = kv.view(B, 2, self.h, C // self.h, -1).transpose(-1, -2).unbind(1)
        o = F.scaled_dot_product_attention(q, k, v)  # B,h,HW,d
        return self.proj(o.transpose(-1, -2).reshape(B, C, H, W))


class GLGBlock(nn.Module):
    def __init__(self, c, heads, mode='gate'):
        super().__init__()
        self.mode = mode  # 'gate' | 'sum' | 'local' | 'global'  (ablations)
        self.norm1 = nn.BatchNorm2d(c)
        self.local = LocalBranch(c) if mode != 'global' else None
        self.glob = GlobalBranch(c, heads) if mode != 'local' else None
        if mode == 'gate':
            self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 4, 1), nn.GELU(), nn.Conv2d(c // 4, c, 1))
        self.norm2 = nn.BatchNorm2d(c)
        self.ffn = nn.Sequential(nn.Conv2d(c, 4 * c, 1), nn.Conv2d(4 * c, 4 * c, 3, 1, 1, groups=4 * c),
                                 nn.GELU(), nn.Conv2d(4 * c, c, 1))
        self.last_gate = None

    def forward(self, x):
        y = self.norm1(x)
        if self.mode == 'local':
            m = self.local(y)
        elif self.mode == 'global':
            m = self.glob(y)
        elif self.mode == 'sum':
            m = self.local(y) + self.glob(y)
        else:
            g = torch.sigmoid(self.gate(y))
            self.last_gate = g.detach()
            m = g * self.local(y) + (1 - g) * self.glob(y)
        x = x + m
        return x + self.ffn(self.norm2(x))


class GLGNet(nn.Module):
    def __init__(self, in_ch, n_cls, size, dims=(64, 128, 256), depths=(2, 2, 4), mode='gate'):
        super().__init__()
        stem_stride = 1 if size <= 32 else 4
        self.stem = nn.Sequential(nn.Conv2d(in_ch, dims[0], 3 if stem_stride == 1 else 4, stem_stride,
                                            1 if stem_stride == 1 else 0), nn.BatchNorm2d(dims[0]), nn.GELU())
        layers = []
        for i, (d, n) in enumerate(zip(dims, depths)):
            if i:
                layers += [nn.BatchNorm2d(dims[i - 1]), nn.Conv2d(dims[i - 1], d, 2, 2)]
            layers += [GLGBlock(d, max(1, d // 32), mode) for _ in range(n)]
        self.stages = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.BatchNorm2d(dims[-1]), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dims[-1], n_cls))

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


def light(name, in_ch, n_cls, size):
    """torchvision lightweight nets; at 28px the first two stride-2 ops become stride 1 (total stride 8, as ResNet)."""
    tv = torchvision.models
    if name == 'mobilenetv2':
        net = tv.mobilenet_v2(num_classes=n_cls)
        stem, down = net.features[0][0], net.features[2].conv[1][0]
    elif name == 'efficientnetb0':
        net = tv.efficientnet_b0(num_classes=n_cls)
        stem, down = net.features[0][0], net.features[2][0].block[1][0]
    elif name == 'shufflenetv2':
        net = tv.shufflenet_v2_x1_0(num_classes=n_cls)
        stem, down = net.conv1[0], None
        if size == 28:
            net.maxpool = nn.Identity()
    new = nn.Conv2d(in_ch, stem.out_channels, stem.kernel_size, 1 if size == 28 else 2, stem.padding, bias=False)
    if name == 'shufflenetv2':
        net.conv1[0] = new
    else:
        net.features[0][0] = new
    if size == 28 and down is not None:
        down.stride = (1, 1)
    return net


def transformer(name, in_ch, n_cls, size):
    """Transformer / hybrid baselines. 28px follows the same rule as the light CNNs (total stride 8 before the last stages)."""
    tv = torchvision.models
    if name == 'vit-tiny':  # DeiT-Ti configuration; 49 tokens at 28px, 64 at 64/128px
        patch = {28: 4, 64: 8}.get(size, 16)
        net = tv.VisionTransformer(image_size=size, patch_size=patch, num_layers=12, num_heads=3, hidden_dim=192,
                                   mlp_dim=768, num_classes=n_cls)
        net.conv_proj = nn.Conv2d(in_ch, 192, patch, patch)
    elif name == 'swin-t':  # torchvision Swin-T; patch 2 at 28px gives 14/7/4/2 token grids, patch 4 otherwise
        patch = 2 if size == 28 else 4
        net = tv.SwinTransformer(patch_size=[patch, patch], embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                                 window_size=[7, 7], stochastic_depth_prob=0.2, num_classes=n_cls)
        net.features[0][0] = nn.Conv2d(in_ch, 96, patch, patch)
    elif name == 'mobilevit-s':  # timm MobileViT-S (hybrid); at 28px stem and first downsampling use stride 1
        import timm
        net = timm.create_model('mobilevit_s', num_classes=n_cls, in_chans=in_ch)
        if size == 28:
            net.stem.conv.stride = (1, 1)
            net.stages[1][0].conv2_kxk.conv.stride = (1, 1)
    return net


def build(name, in_ch, n_cls, size):
    if name in ('vit-tiny', 'swin-t', 'mobilevit-s'):
        return transformer(name, in_ch, n_cls, size)
    if name in ('mobilenetv2', 'efficientnetb0', 'shufflenetv2'):
        return light(name, in_ch, n_cls, size)
    if name == 'resnet18':
        return resnet(18, in_ch, n_cls, size)
    if name == 'resnet50':
        return resnet(50, in_ch, n_cls, size)
    if name.startswith('glgnet'):  # glgnet | glgnet-sum | glgnet-local | glgnet-global
        return GLGNet(in_ch, n_cls, size, mode=name.split('-')[1] if '-' in name else 'gate')
    raise ValueError(name)


if __name__ == '__main__':  # shape/param self-check
    for m in ('glgnet', 'glgnet-sum', 'glgnet-local', 'glgnet-global', 'resnet18',
              'mobilenetv2', 'efficientnetb0', 'shufflenetv2', 'vit-tiny', 'swin-t', 'mobilevit-s'):
        for s in (28, 64, 128):
            net = build(m, 1 if s == 64 else 3, 9, s)
            out = net(torch.randn(2, 1 if s == 64 else 3, s, s))
            assert out.shape == (2, 9), out.shape
        print(m, f'{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M')
