import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ---------------------------------------------------------------------------
#  SPADE normalization
# ---------------------------------------------------------------------------

class SPADE(nn.Module):
    """Spatially-Adaptive (DE)normalization conditioned on a label map."""

    def __init__(self, norm_nc, label_nc, nhidden=64):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Conv2d(nhidden, norm_nc, 3, 1, 1)
        self.beta = nn.Conv2d(nhidden, norm_nc, 3, 1, 1)

    def forward(self, x, seg):
        normed = self.param_free_norm(x)
        seg = F.interpolate(seg, size=x.shape[2:], mode="bilinear", align_corners=False)
        shared = self.shared(seg)
        return normed * (1 + self.gamma(shared)) + self.beta(shared)


# ---------------------------------------------------------------------------
#  SPADE residual block
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Scaled dot-product spatial self-attention."""

    def __init__(self, in_nc):
        super().__init__()
        self.ch_k = max(in_nc // 8, 1)
        self.query = spectral_norm(nn.Conv2d(in_nc, self.ch_k, 1))
        self.key = spectral_norm(nn.Conv2d(in_nc, self.ch_k, 1))
        self.value = spectral_norm(nn.Conv2d(in_nc, in_nc, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = self.ch_k ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = torch.bmm(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class SPADEResBlock(nn.Module):
    def __init__(self, fin, fout, label_nc):
        super().__init__()
        fmid = min(fin, fout)
        self.learned_shortcut = (fin != fout)

        self.norm0 = SPADE(fin, label_nc)
        self.conv0 = nn.Conv2d(fin, fmid, 3, 1, 1)
        self.norm1 = SPADE(fmid, label_nc)
        self.conv1 = nn.Conv2d(fmid, fout, 3, 1, 1)

        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)
            self.conv_s = nn.Conv2d(fin, fout, 1, bias=False)

    def forward(self, x, seg):
        shortcut = x
        if self.learned_shortcut:
            shortcut = self.conv_s(self.norm_s(shortcut, seg))
        out = self.conv0(F.leaky_relu(self.norm0(x, seg), 0.2))
        out = self.conv1(F.leaky_relu(self.norm1(out, seg), 0.2))
        return shortcut + out


# ---------------------------------------------------------------------------
#  SPADE Generator  (fully convolutional — no FC bottleneck)
# ---------------------------------------------------------------------------

class SPADEGenerator(nn.Module):
    """
    Fully-convolutional SPADE generator for heightmap regression.
    Encoder downsamples the label spatially (preserving structure).
    Decoder upsamples with SPADE residual blocks conditioned on the label.
    No FC layer => dramatically fewer parameters.
    """

    def __init__(self, label_nc=1, output_nc=1, ngf=64, use_attention=True,
                 output_act="sigmoid"):
        super().__init__()
        nf = ngf

        self.enc1 = nn.Sequential(nn.Conv2d(label_nc, nf, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.enc2 = nn.Sequential(nn.Conv2d(nf, nf * 2, 4, 2, 1), nn.InstanceNorm2d(nf * 2), nn.LeakyReLU(0.2, True))
        self.enc3 = nn.Sequential(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1), nn.InstanceNorm2d(nf * 4), nn.LeakyReLU(0.2, True))
        self.enc4 = nn.Sequential(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1), nn.InstanceNorm2d(nf * 8), nn.LeakyReLU(0.2, True))

        self.bottleneck = SPADEResBlock(nf * 8, nf * 8, label_nc)
        self.attn = SelfAttention(nf * 8) if use_attention else nn.Identity()

        # Decoder: 16 -> 32 -> 64 -> 128 -> 256
        self.up4 = SPADEResBlock(nf * 8, nf * 4, label_nc)  # + skip from enc3
        self.up3 = SPADEResBlock(nf * 4, nf * 2, label_nc)  # + skip from enc2
        self.up2 = SPADEResBlock(nf * 2, nf, label_nc)      # + skip from enc1
        self.up1 = SPADEResBlock(nf, nf, label_nc)

        # Skip-connection projection (concatenation -> reduce channels)
        self.skip4 = nn.Conv2d(nf * 8 + nf * 4, nf * 8, 1)
        self.skip3 = nn.Conv2d(nf * 4 + nf * 2, nf * 4, 1)
        self.skip2 = nn.Conv2d(nf * 2 + nf, nf * 2, 1)

        act_map = {
            "sigmoid": nn.Sigmoid(),
            "hardtanh": nn.Hardtanh(0.0, 1.0),
        }
        self.conv_out = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, output_nc, 3, 1, 1),
            act_map.get(output_act, nn.Sigmoid()),
        )

    def forward(self, seg):
        # Encode
        e1 = self.enc1(seg)   # nf,   128
        e2 = self.enc2(e1)    # nf*2, 64
        e3 = self.enc3(e2)    # nf*4, 32
        e4 = self.enc4(e3)    # nf*8, 16

        x = self.bottleneck(e4, seg)
        x = self.attn(x)

        # Decode with skip connections
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 32
        x = self.skip4(torch.cat([x, e3], dim=1))
        x = self.up4(x, seg)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 64
        x = self.skip3(torch.cat([x, e2], dim=1))
        x = self.up3(x, seg)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 128
        x = self.skip2(torch.cat([x, e1], dim=1))
        x = self.up2(x, seg)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 256
        x = self.up1(x, seg)

        return self.conv_out(x)


# ---------------------------------------------------------------------------
#  PatchGAN Discriminator
# ---------------------------------------------------------------------------

class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator with spectral norm; returns per-block features."""

    def __init__(self, input_nc=2, ndf=64, n_layers=3):
        super().__init__()
        self.blocks = nn.ModuleList()

        self.blocks.append(nn.Sequential(
            spectral_norm(nn.Conv2d(input_nc, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),
        ))

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.blocks.append(nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1)),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.blocks.append(nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1)),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ))

        self.blocks.append(nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)),
        ))

    def forward(self, label, image):
        x = torch.cat([label, image], dim=1)
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats


# ---------------------------------------------------------------------------
#  Multi-scale discriminator
# ---------------------------------------------------------------------------

class MultiscaleDiscriminator(nn.Module):
    """Returns list[list[Tensor]]: outer = per-scale, inner = per-block features."""

    def __init__(self, input_nc=2, ndf=64, n_layers=3, num_D=2):
        super().__init__()
        self.discriminators = nn.ModuleList()
        for _ in range(num_D):
            self.discriminators.append(NLayerDiscriminator(input_nc, ndf, n_layers))
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, label, image):
        results = []
        for disc in self.discriminators:
            results.append(disc(label, image))
            label = self.downsample(label)
            image = self.downsample(image)
        return results
