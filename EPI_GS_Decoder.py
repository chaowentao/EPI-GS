"""
EPI-GS Decoder v4-freq-gs2lr-occ5-dfreq: COMBINATION of occ5 + dfreq.

Two orthogonal improvements stacked on the gs2lr base decoder:

  (1) occ5  -- Occlusion-Aware 5-Point EPI sampling
        delta_ang = {0, +/-1, +/-2}; weight = w_ang * w_epi * w_occ with
        w_occ = exp(-occ_gamma * |d_center - d_neighbor|). The +/-2 samples
        are softly gated by the angular Gaussian and protected by w_occ.

  (2) dfreq -- Disparity-Guided Frequency Enhancement (Option B)
        FrequencyEnhancement uses the predicted disparity d_hr as an extra
        Fourier modulation axis: freq += freq_disp_conv(fused) * (d_scale*d_hr),
        mirroring the EPI slope-frequency duality.

These touch different modules (EPI sampler vs freq output), so they combine
cleanly. Decoder ~136K params (occ5 132K + freq_disp_conv ~4.7K).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================
# Basic modules
# ============================================================

class MLPRefiner(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list=None):
        super().__init__()
        layers = []
        lastv = in_dim
        if hidden_list:
            for hidden in hidden_list:
                layers.append(nn.Linear(lastv, hidden))
                layers.append(nn.ReLU(inplace=True))
                lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.reshape(-1, x.shape[-1]))
        return x.view(*shape, -1)


def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret



# ============================================================
# Spatial implicit sampling (NO Local Ensemble - single nearest)
# ============================================================

def spatial_implicit_sampling(fea, HH, WW):
    b, c, h, w = fea.shape
    device = fea.device
    coord_hr = make_coord((HH, WW)).repeat(b, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).to(device)
    feat_coord = make_coord((h, w), flatten=False).to(device)
    feat_coord = feat_coord.permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)
    hr_fea = F.grid_sample(fea, coord_hr.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)
    hr_fea = hr_fea[:, :, 0, :].permute(0, 2, 1)
    hr_coord = F.grid_sample(feat_coord, coord_hr.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)
    hr_coord = hr_coord[:, :, 0, :].permute(0, 2, 1)
    rel_coord = coord_hr - hr_coord
    rel_coord[:, :, 0] *= h
    rel_coord[:, :, 1] *= w
    cell = torch.ones_like(rel_coord)
    cell[:, :, 0] *= 2 / HH * h
    cell[:, :, 1] *= 2 / WW * w
    return torch.cat([hr_fea, rel_coord, cell], dim=-1)


# ============================================================
# Spatial implicit sampling (with Local Ensemble)
# ============================================================

def spatial_implicit_sampling_le(fea, HH, WW):
    b, c, h, w = fea.shape
    device = fea.device
    eps_shift = 1e-6
    rx = 2.0 / h / 2
    ry = 2.0 / w / 2
    coord_hr = make_coord((HH, WW)).repeat(b, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).to(device)
    feat_coord = make_coord((h, w), flatten=False).to(device)
    feat_coord = feat_coord.permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)
    vx_lst, vy_lst = [-1, 1], [-1, 1]
    inp_list, areas = [], []
    for vx in vx_lst:
        for vy in vy_lst:
            coord_ = coord_hr.clone()
            coord_[:, :, 0] += vx * rx + eps_shift
            coord_[:, :, 1] += vy * ry + eps_shift
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
            q_feat = F.grid_sample(fea, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)
            q_feat = q_feat[:, :, 0, :].permute(0, 2, 1)
            q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)
            q_coord = q_coord[:, :, 0, :].permute(0, 2, 1)
            rel_coord = coord_hr - q_coord
            rel_coord[:, :, 0] *= h
            rel_coord[:, :, 1] *= w
            cell = torch.ones_like(rel_coord)
            cell[:, :, 0] *= 2 / HH * h
            cell[:, :, 1] *= 2 / WW * w
            inp = torch.cat([q_feat, rel_coord, cell], dim=-1)
            inp_list.append(inp)
            area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
            areas.append(area + 1e-9)
    return inp_list, areas


# ============================================================
# EPI-GS2-LR-OCC5: Per-LR-pixel GS + Occlusion-Aware 5-Point
# ============================================================

def epi_gs2lr_occ5_sampling(fea, Ang, Spa, gs_mlp, occ_gamma=3.0):
    """Per-LR-pixel GS sampling: 5-point {0,+/-1,+/-2} + occlusion-aware.

    Args:
        fea: (B, C, ang, spa) EPI feature map (LR)
        Ang: HR angular resolution
        Spa: HR spatial resolution
        gs_mlp: MLPRefiner (C -> 3)
        occ_gamma: occlusion decay rate (higher = stricter)

    Returns:
        out_feat: (B, Ang*Spa, C)
        d_map_lr: (B, ang, spa)
    """
    b, c, ang, spa = fea.shape
    device = fea.device
    N_query = Ang * Spa
    delta_ang_list = [0, 1, -1, 2, -2]

    # Step 1: Predict Gaussian params at LR resolution
    fea_perm = fea.permute(0, 2, 3, 1)
    gs_out = gs_mlp(fea_perm)
    L11_map = torch.clamp(F.softplus(gs_out[:, :, :, 0:1]), min=0.1, max=5.0)
    L21_map = gs_out[:, :, :, 1:2]
    L22_map = torch.clamp(F.softplus(gs_out[:, :, :, 2:3]), min=0.1, max=5.0)
    d_map_lr = (L21_map / (L11_map + 1e-8)).squeeze(-1)

    L11_fm = L11_map.permute(0, 3, 1, 2)
    L22_fm = L22_map.permute(0, 3, 1, 2)
    d_fm = d_map_lr.unsqueeze(1)

    # Step 2: Sample params at HR positions
    coord_hr = make_coord((Ang, Spa)).repeat(b, 1, 1).clamp(-1 + 1e-6, 1 - 1e-6).to(device)
    sampling_grid = coord_hr.flip(-1).unsqueeze(1)
    L11_hr = F.grid_sample(L11_fm, sampling_grid, mode='nearest', align_corners=False)[:, 0, 0, :]
    L22_hr = F.grid_sample(L22_fm, sampling_grid, mode='nearest', align_corners=False)[:, 0, 0, :]
    d_hr = F.grid_sample(d_fm, sampling_grid, mode='nearest', align_corners=False)[:, 0, 0, :]

    # Step 3: Gaussian-weighted sampling with occlusion awareness
    query_spa_norm = coord_hr[:, :, 1]
    query_spa_pixel = (query_spa_norm + 1.0) * spa / 2.0 - 0.5

    q_feat = F.grid_sample(fea, sampling_grid, mode='nearest', align_corners=False)
    q_feat = q_feat[:, :, 0, :].permute(0, 2, 1)

    weighted_sum = torch.zeros(b, N_query, c, device=device)
    weight_sum = torch.zeros(b, N_query, 1, device=device)

    for delta_ang in delta_ang_list:
        if delta_ang == 0:
            high_fea = q_feat
            w = torch.ones(b, N_query, 1, device=device)
        else:
            ideal_pixel = query_spa_pixel + d_hr * delta_ang
            snapped_pixel = torch.round(ideal_pixel)
            residual = snapped_pixel - ideal_pixel

            snapped_norm = (snapped_pixel + 0.5) * 2.0 / spa - 1.0
            snapped_norm = snapped_norm.clamp(-1 + 1e-6, 1 - 1e-6)
            ang_norm = coord_hr[:, :, 0] + delta_ang * (2.0 / ang)
            ang_norm = ang_norm.clamp(-1 + 1e-6, 1 - 1e-6)

            sample_coord = torch.stack([snapped_norm, ang_norm], dim=-1)
            sampled = F.grid_sample(fea, sample_coord.unsqueeze(1), mode='nearest', align_corners=False)
            high_fea = sampled[:, :, 0, :].permute(0, 2, 1)

            # Gaussian weights
            w_ang = torch.exp(-float(delta_ang ** 2) / (2.0 * L11_hr ** 2 + 1e-8))
            w_epi = torch.exp(-residual ** 2 / (2.0 * L22_hr ** 2 + 1e-8))

            # Occlusion-aware weight
            d_neighbor = F.grid_sample(d_fm, sample_coord.unsqueeze(1), mode='nearest', align_corners=False)
            d_neighbor = d_neighbor[:, 0, 0, :]
            w_occ = torch.exp(-occ_gamma * (d_hr - d_neighbor).abs())

            w = (w_ang * w_epi * w_occ).unsqueeze(-1)

        weighted_sum = weighted_sum + w * high_fea
        weight_sum = weight_sum + w

    out_feat = weighted_sum / (weight_sum + 1e-8)
    return out_feat, d_map_lr


# ============================================================
# Frequency Enhancement Module (dfreq: disparity-guided, Option B)
# ============================================================

class FrequencyEnhancement(nn.Module):
    """GaussianSR-style frequency modulation with disparity-guided axis.

    Besides the spatial relative-coordinate modulation, the predicted
    disparity d is used as an additional Fourier modulation axis. This mirrors
    the EPI slope-frequency duality, where a surface of disparity d concentrates
    EPI energy along a frequency orientation proportional to d.
    """

    def __init__(self, channels, use_freq=True, d_scale=1.0):
        super().__init__()
        self.C = channels
        self.use_freq = use_freq
        self.d_scale = d_scale
        if use_freq:
            self.coef_conv = nn.Conv2d(channels, channels, 3, padding=1)
            self.freq_conv = nn.Conv2d(channels, channels, 3, padding=1)
            # Disparity modulation axis: produces C//2 channels to match freq
            self.freq_disp_conv = nn.Conv2d(channels, channels // 2, 3, padding=1)
            self.phase = nn.Linear(2, channels // 2, bias=False)
            self.out_conv = nn.Conv2d(channels, 1, 1, 1, 0)
        else:
            self.out_conv = nn.Conv2d(channels, 1, 1, 1, 0)

    def _make_rel_coord_grid(self, HH, WW, h, w, device):
        coord_hr = make_coord((HH, WW)).unsqueeze(0).to(device)
        feat_coord = make_coord((h, w), flatten=False).to(device)
        feat_coord = feat_coord.permute(2, 0, 1).unsqueeze(0)
        q_coord = F.grid_sample(feat_coord, coord_hr.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)
        q_coord = q_coord[:, :, 0, :].permute(0, 2, 1)
        rel_coord = coord_hr - q_coord
        rel_coord[:, :, 0] *= h
        rel_coord[:, :, 1] *= w
        return rel_coord.view(1, HH, WW, 2).permute(0, 3, 1, 2)

    def forward(self, fused_feat, h_lr, w_lr, scale, d_hr=None):
        if not self.use_freq:
            return self.out_conv(fused_feat)
        B_total, C, HH, WW = fused_feat.shape
        device = fused_feat.device
        coef = self.coef_conv(fused_feat)
        freq = self.freq_conv(fused_feat)
        rel_coord = self._make_rel_coord_grid(HH, WW, h_lr, w_lr, device)
        freq = freq.view(B_total, 2, C // 2, HH, WW)
        freq = (freq * rel_coord.unsqueeze(2)).sum(dim=1)

        # Disparity as an additional Fourier modulation axis
        if d_hr is not None:
            freq_disp = self.freq_disp_conv(fused_feat)
            freq = freq + freq_disp * (self.d_scale * d_hr)

        cell = torch.tensor([2.0 / HH * h_lr, 2.0 / WW * w_lr], device=device)
        phase = self.phase(cell)
        freq = freq + phase.view(1, -1, 1, 1)
        freq = torch.cat([torch.cos(np.pi * freq), torch.sin(np.pi * freq)], dim=1)
        return self.out_conv(coef * freq)


# ============================================================
# Main decoder
# ============================================================

class LF_Decoding(nn.Module):
    def __init__(self, angRes, in_channels=64, internal_channels=32,
                 hidden_dim=128, n_hidden=2, use_freq=True,
                 occ_gamma=3.0, d_scale=1.0):
        super().__init__()
        self.angRes = angRes
        self.C = internal_channels
        self.occ_gamma = occ_gamma
        self.shrink = nn.Conv2d(in_channels, internal_channels, 1, 1, 0)
        C = internal_channels
        self.spatial_mlp = MLPRefiner(C + 4, C, hidden_list=[hidden_dim] * n_hidden)
        gs_hidden_list = [256, 128]
        self.gs_mlp_h = MLPRefiner(C, 3, hidden_list=gs_hidden_list)
        self.gs_mlp_v = MLPRefiner(C, 3, hidden_list=gs_hidden_list)
        self.fusion = nn.Sequential(nn.Conv2d(C * 2, C, 1, 1, 0), nn.ReLU(inplace=True))
        self.freq_enhance = FrequencyEnhancement(C, use_freq=use_freq, d_scale=d_scale)

    def forward(self, fea, HH, WW, scale):
        fea = self.shrink(fea)
        B_total, C, h, w = fea.shape
        B = B_total // (self.angRes ** 2)

        sp_inp = spatial_implicit_sampling(fea, HH, WW)
        sp_fea = self.spatial_mlp(sp_inp)
        sp_fea = rearrange(sp_fea, '(b n) (h w) c -> (b n) c h w', b=B, n=self.angRes**2, h=HH, w=WW)

        fea_h = rearrange(fea, '(b a1 a2) c h w -> (b a1 h) c a2 w', a1=self.angRes, a2=self.angRes, h=h, w=w)
        epi_fea_h, d_h = epi_gs2lr_occ5_sampling(fea_h, self.angRes, WW, self.gs_mlp_h, self.occ_gamma)
        epi_fea_h = rearrange(epi_fea_h, '(b a1 h) (a2 w) c -> (b a2 w) c a1 h',
                              a1=self.angRes, a2=self.angRes, h=h, w=WW)
        epi_fea_v, d_v = epi_gs2lr_occ5_sampling(epi_fea_h, self.angRes, HH, self.gs_mlp_v, self.occ_gamma)
        epi_fea_v = rearrange(epi_fea_v, '(b a2 w) (a1 h) c -> (b a1 a2) c h w',
                              a1=self.angRes, a2=self.angRes, h=HH, w=WW)
        d_map = rearrange(d_h, '(b a1 h) a2 w -> b (a1 a2) h w', b=B, a1=self.angRes, a2=self.angRes, h=h, w=w)

        fused = torch.cat([sp_fea, epi_fea_v], dim=1)
        fused_feat = self.fusion(fused)

        # Upsample LR disparity to HR for disparity-guided freq modulation
        d_hr = d_map.view(B_total, 1, h, w)
        d_hr = F.interpolate(d_hr, size=(HH, WW), mode='nearest')

        out = self.freq_enhance(fused_feat, h, w, scale, d_hr=d_hr)
        out = rearrange(out, '(b n) c h w -> b n c h w', b=B, n=self.angRes**2)
        return out, d_map


# ============================================================
# Full network
# ============================================================

class Net(nn.Module):
    def __init__(self, angRes, in_channels=64, internal_channels=32,
                 hidden_dim=128, n_hidden=2, use_freq=True,
                 occ_gamma=3.0, d_scale=1.0):
        super().__init__()
        self.angRes = angRes
        n_groups, n_blocks, channels = 4, 4, 64
        self.angFE = nn.Conv2d(1, channels, kernel_size=angRes, stride=angRes, padding=0, bias=False)
        self.spaFE = nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False)
        self.interaction = Cascaded_InterGroups(angRes, n_groups, n_blocks, channels)
        self.bottleneck = BottleNeck(angRes, n_groups, channels)
        self.upsample = LF_Decoding(angRes, in_channels=channels, internal_channels=internal_channels,
                                    hidden_dim=hidden_dim, n_hidden=n_hidden, use_freq=use_freq,
                                    occ_gamma=occ_gamma, d_scale=d_scale)

    def forward(self, x, HH, WW, scale):
        lr_upscale = interpolate(x, self.angRes, scale_size=(HH, WW), mode='bicubic')
        x = SAI2MacPI(x, self.angRes)
        xa, xs = self.angFE(x), self.spaFE(x)
        buffer_a, buffer_s = self.interaction(xa, xs)
        buffer_out = self.bottleneck(buffer_a, buffer_s) + xs
        fea = MacPI2SAI(buffer_out, self.angRes)
        fea = LFsplit(fea, self.angRes)
        b, n, c, h, w = fea.size()
        fea = fea.contiguous().view(b * n, c, h, w)
        out, d_map = self.upsample(fea, HH, WW, scale)
        out = rearrange(out, 'b (a1 a2) c h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        return out + lr_upscale, d_map


# ============================================================
# Encoder sub-modules
# ============================================================

class InterBlock(nn.Module):
    def __init__(self, angRes, channels):
        super().__init__()
        self.Spa2Ang = nn.Conv2d(channels, channels, kernel_size=angRes, stride=angRes, padding=0, bias=False)
        self.Ang2Spa = nn.Sequential(
            nn.Conv2d(channels, angRes*angRes*channels, 1, 1, 0, bias=False), nn.PixelShuffle(angRes))
        self.AngConv = nn.Conv2d(2*channels, channels, 1, 1, 0, bias=False)
        self.SpaConv = nn.Conv2d(2*channels, channels, 3, 1, dilation=angRes, padding=angRes, bias=False)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, xa, xs):
        buffer_ang2 = self.ReLU(self.Spa2Ang(xs))
        buffer_spa2 = self.Ang2Spa(xa)
        out_a = self.ReLU(self.AngConv(torch.cat((xa, buffer_ang2), 1))) + xa
        out_s = self.ReLU(self.SpaConv(torch.cat((xs, buffer_spa2), 1))) + xs
        return out_a, out_s


class InterGroup(nn.Module):
    def __init__(self, angRes, n_block, channels):
        super().__init__()
        self.n_block = n_block
        self.chained_blocks = nn.Sequential(*[InterBlock(angRes, channels) for _ in range(n_block)])

    def forward(self, xa, xs):
        for i in range(self.n_block):
            xa, xs = self.chained_blocks[i](xa, xs)
        return xa, xs


class Cascaded_InterGroups(nn.Module):
    def __init__(self, angRes, n_group, n_block, channels):
        super().__init__()
        self.n_group = n_group
        self.body = nn.Sequential(*[InterGroup(angRes, n_block, channels) for _ in range(n_group)])

    def forward(self, buffer_a, buffer_s):
        out_a, out_s = [], []
        for i in range(self.n_group):
            buffer_a, buffer_s = self.body[i](buffer_a, buffer_s)
            out_a.append(buffer_a)
            out_s.append(buffer_s)
        return torch.cat(out_a, 1), torch.cat(out_s, 1)


class BottleNeck(nn.Module):
    def __init__(self, angRes, n_blocks, channels):
        super().__init__()
        self.AngBottle = nn.Conv2d(n_blocks*channels, channels, 1, 1, 0, bias=False)
        self.Ang2Spa = nn.Sequential(
            nn.Conv2d(channels, angRes*angRes*channels, 1, 1, 0, bias=False), nn.PixelShuffle(angRes))
        self.SpaBottle = nn.Conv2d((n_blocks+1)*channels, channels, 3, 1, dilation=angRes, padding=angRes, bias=False)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, xa, xs):
        xa = self.ReLU(self.AngBottle(xa))
        xs = torch.cat((xs, self.Ang2Spa(xa)), 1)
        return self.ReLU(self.SpaBottle(xs))


# ============================================================
# Utilities
# ============================================================

def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    return torch.cat(out, 2)

def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    return torch.cat(tempU, dim=2)

def interpolate(x, angRes, scale_size, mode):
    B, _, H, W = x.size()
    h, w = H // angRes, W // angRes
    x_up = x.view(B, 1, angRes, h, angRes, w).permute(0, 2, 4, 1, 3, 5).contiguous().view(B*angRes**2, 1, h, w)
    x_up = F.interpolate(x_up, size=scale_size, mode=mode, align_corners=False)
    x_up = x_up.view(B, angRes, angRes, 1, scale_size[0], scale_size[1])
    return x_up.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, 1, scale_size[0]*angRes, scale_size[1]*angRes)

def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h, w = H // angRes, W // angRes
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])
    return torch.stack(data_sv, dim=1)

def compute_lr_warp_loss(x, d_map, angRes):
    B = x.shape[0]
    uh, vw = x.shape[2], x.shape[3]
    h, w = uh // angRes, vw // angRes
    u_center, v_center = angRes // 2, angRes // 2
    views = x.view(B, angRes, h, angRes, w).permute(0, 1, 3, 2, 4)
    center_view = views[:, u_center, v_center]
    d_views = d_map.view(B, angRes, angRes, h, w)
    ys = torch.linspace(-1, 1, h, device=x.device)
    xs = torch.linspace(-1, 1, w, device=x.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1)
    warp_loss = torch.tensor(0.0, device=x.device)
    n_views = 0
    for u in range(angRes):
        for v in range(angRes):
            if u == u_center and v == v_center:
                continue
            du, dv = u - u_center, v - v_center
            d = d_views[:, u, v]
            new_grid = grid.unsqueeze(0).expand(B, -1, -1, -1).clone()
            new_grid[:, :, :, 0] += dv * d * (2.0 / w)
            new_grid[:, :, :, 1] += du * d * (2.0 / h)
            warped = F.grid_sample(views[:, u, v].unsqueeze(1), new_grid,
                                   mode='bilinear', padding_mode='border', align_corners=False)
            warp_loss += F.l1_loss(warped.squeeze(1), center_view)
            n_views += 1
    return warp_loss / n_views


# ============================================================
# Test
# ============================================================

if __name__ == '__main__':
    angRes = 5
    print("=" * 60)
    print("v4-freq-gs2lr-occ5-dfreq: Occlusion-5pt + Disparity-Guided Freq")
    print("=" * 60)
    net = Net(angRes, internal_channels=32, hidden_dim=128,
              use_freq=True, occ_gamma=3.0, d_scale=1.0).cuda()

    total = sum(p.nelement() for p in net.parameters())
    enc_p = sum(p.nelement() for n, p in net.named_parameters() if not n.startswith('upsample'))
    dec_p = sum(p.nelement() for n, p in net.named_parameters() if n.startswith('upsample'))
    gs_p = sum(p.nelement() for n, p in net.named_parameters() if 'gs_mlp' in n)
    freq_p = sum(p.nelement() for n, p in net.named_parameters() if 'freq_enhance' in n)
    freq_disp_p = sum(p.nelement() for n, p in net.named_parameters() if 'freq_disp_conv' in n)

    print(f'Total: {total/1e6:.3f}M | Encoder: {enc_p/1e6:.3f}M | Decoder: {dec_p/1e3:.1f}K')
    print(f'  gs_mlp (H+V): {gs_p/1e3:.1f}K')
    print(f'  freq_enhance: {freq_p/1e3:.1f}K  (freq_disp_conv: {freq_disp_p/1e3:.1f}K)')

    x = torch.randn(1, 1, 32*angRes, 32*angRes).cuda()
    out, d_map = net(x, 64, 64, 2)
    print(f'\nInput: {x.shape} -> Output: {out.shape} | d_map: {d_map.shape}')
    assert out.shape == (1, 1, 64*angRes, 64*angRes)
    assert d_map.shape == (1, 25, 32, 32)
    print('Forward pass OK!')

    loss = F.l1_loss(out, torch.randn_like(out)) + 0.1 * compute_lr_warp_loss(x, d_map, angRes)
    loss.backward()
    print(f'Loss: {loss.item():.6f} | Backward OK!')
