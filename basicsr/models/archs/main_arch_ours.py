import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm

from basicsr.models.archs.arch_util_ours import (
    MixBlock,
    Mutual_Attention,
    ResBlock,
    SAM,
    conv,
    conv3x3,
    conv_down,
)


def scer_to_voxel_general(event: torch.Tensor) -> torch.Tensor:
    assert event.dim() == 4, "event must be (B, C, H, W)"
    _, c, _, _ = event.shape
    assert c >= 2, "C must be >= 2"

    mid_left = c // 2 - 1
    mid_right = c // 2
    edge = torch.zeros_like(event)

    if mid_left > 0:
        edge[:, :mid_left] = event[:, :mid_left] - event[:, 1:mid_left + 1]

    edge[:, mid_left:mid_right + 1] = event[:, mid_left:mid_right + 1]

    if mid_right + 1 < c:
        edge[:, mid_right + 1:] = event[:, mid_right + 1:] - event[:, mid_right:-1]

    return edge


def _feature_flow_costvolume(source_feat, target_feat, radius=3, inner_radius=0, temperature=1.0):
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, but got {temperature}")

    disps = []
    costs = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if inner_radius > 0 and max(abs(dx), abs(dy)) <= inner_radius:
                continue
            shifted = torch.roll(target_feat, shifts=(dy, dx), dims=(2, 3))
            cost = (source_feat * shifted).mean(dim=1, keepdim=True)
            costs.append(cost)
            disps.append((dx, dy))

    if not costs:
        raise ValueError("No displacement candidates were generated for the given radius and inner_radius.")

    cost_volume = torch.cat(costs, dim=1)
    prob = F.softmax(cost_volume / temperature, dim=1)
    disp = source_feat.new_tensor(disps).view(1, -1, 2, 1, 1)
    flow_x = (prob * disp[:, :, 0]).sum(dim=1, keepdim=True)
    flow_y = (prob * disp[:, :, 1]).sum(dim=1, keepdim=True)
    return torch.cat([flow_x, flow_y], dim=1)


def _build_symmetric_ev_flow_feature(
    event,
    ev_flow_stem,
    ev_flow_downsample,
    radius=3,
    inner_radius=0,
    temperature=0.1,
):
    if event.ndim != 4 or event.shape[1] != 8:
        raise ValueError(f"Expected event shape (B,8,H,W), but got {tuple(event.shape)}")

    if ev_flow_downsample > 1:
        event = F.interpolate(
            event,
            scale_factor=1 / ev_flow_downsample,
            mode="bilinear",
            align_corners=False,
        )

    pairs = [
        (0, 1),
        (1, 2),
        (2, 3),
        (5, 4),
        (6, 5),
        (7, 6),
    ]
    flow_list = []
    for src, tgt in pairs:
        flow_list.append(
            _feature_flow_costvolume(
                event[:, tgt:tgt + 1, :, :],
                event[:, src:src + 1, :, :],
                radius=radius,
                inner_radius=inner_radius,
                temperature=temperature,
            )
        )

    flow = torch.stack(flow_list, dim=1)
    b, num_pairs, pair_channels, h, w = flow.shape
    flow_feat = flow.reshape(b, num_pairs * pair_channels, h, w)
    if ev_flow_downsample > 1:
        flow_feat = F.interpolate(
            flow_feat,
            scale_factor=ev_flow_downsample,
            mode="bilinear",
            align_corners=False,
        )
    return ev_flow_stem(flow_feat), flow


class ChannelRefineBlock(nn.Module):
    def __init__(self, feats, distillation_rate=0.5):
        super(ChannelRefineBlock, self).__init__()
        self.distilled_channels = int(feats // 2 * distillation_rate)
        self.remaining_channels = int(feats // 2 - self.distilled_channels)
        self.before_refine = nn.Conv2d(feats, feats, 1, 1, 0)
        self.c1 = weight_norm(nn.Conv2d(feats, feats // 2, 3, 1, padding=1, groups=1, bias=False))
        self.c2 = weight_norm(nn.Conv2d(self.remaining_channels, feats // 2, 3, 1, padding=1, groups=1, bias=False))
        self.c3 = weight_norm(
            nn.Conv2d(self.remaining_channels, self.distilled_channels, 3, 1, padding=1, groups=1, bias=False)
        )
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.c5 = weight_norm(
            nn.Conv2d(self.distilled_channels * 3, feats, 3, 1, padding=1, groups=1, bias=False)
        )
        self.after_refine = nn.Conv2d(feats, feats, 1, 1, 0)

    def forward(self, input):
        input = self.before_refine(input)
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        out = torch.cat([distilled_c1, distilled_c2, out_c3], dim=1)
        out_fused = self.c5(out) + input
        out_fused = self.after_refine(out_fused)
        return out_fused


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super(SEBlock, self).__init__()
        hidden = max(channels // reduction, 1)
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), MixBlock(channels, out_channels=channels, hidden_channels=hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class _Memory_Block_prompt(nn.Module):
    def __init__(self, hdim, kdim, moving_average_rate=0.999, pool_kernel=7):
        super().__init__()
        self.c = hdim
        self.k = kdim
        self.moving_average_rate = moving_average_rate
        self.pool_kernel = pool_kernel
        
        # EMA memory should be updated in-place, not re-registered as a Parameter.
        self.register_buffer("units", torch.randn(kdim, hdim))
        self.prompts = nn.Parameter(torch.rand(kdim, hdim), requires_grad=True)
        self.training=True
#         self.event_proj = nn.Conv2d(hdim, kdim,1,1,0)
        self.motion_gate_proj = nn.Sequential(
            nn.Conv2d(6, hdim, 1, 1, 0),
            nn.Sigmoid()
        )
    def update(self, x, score):
        m = self.units
        x = x.detach()
        embed_ind = torch.max(score, dim=1)[1]
        embed_onehot = F.one_hot(embed_ind, self.k).type(x.dtype)
        embed_onehot_sum = embed_onehot.sum(0)
        embed_sum = x.transpose(0, 1) @ embed_onehot
        embed_mean = embed_sum / (embed_onehot_sum + 1e-6)
        new_data = m * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)
        if self.training:
            with torch.no_grad():
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(new_data, op=dist.ReduceOp.SUM)
                    new_data /= dist.get_world_size()
                self.units.copy_(new_data)
        return self.units

    def forward(self, x, raw_flow, update_flag=True):
        b, c, h, w = x.size()
        assert c == self.c

#         event_weight = self.event_proj(y)
#         event_weight = event_weight.permute(0, 2, 3, 1).reshape(-1, self.k)
        # motion magnitude map
        motion_mag = torch.norm(raw_flow, p=2, dim=2) # (B, 6, H, W)
        # local maximum motion magnitude
        if motion_mag.shape[-2:] != x.shape[-2:]:
            motion_mag = F.interpolate(
                motion_mag,
                size=x.shape[-2:],
                mode="nearest"
            )

        motion_mag = F.max_pool2d(
            motion_mag,
            kernel_size=self.pool_kernel,
            stride=1,
            padding=self.pool_kernel // 2
        )
        # normalize per sample
        mag_max = motion_mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        motion_mag = motion_mag / mag_max

        motion_gate = self.motion_gate_proj(motion_mag)  # (B,hdim,H,W)
        
        # memory addressing by motion feature direction
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)

        m = self.units
        p = self.prompts
        xn = F.normalize(x_flat, dim=1)
        mn = F.normalize(m, dim=1)
        score = torch.matmul(xn, mn.t())

        if update_flag:
            m = self.update(x_flat, score)
            mn = F.normalize(m, dim=1)
            score = torch.matmul(xn, mn.t())

        # if event.shape[1] != score.shape[1]:
        #     event = event.mean(dim=1, keepdim=True)
#         score = score + event_weight
        # score = score * event
        soft_label = F.softmax(score, dim=1)
        out = torch.matmul(soft_label, p).view(b, h, w, c).permute(0, 3, 1, 2)
        out_m = torch.matmul(soft_label, m).view(b, h, w, c).permute(0, 3, 1, 2)
        
        # motion-magnitude-aware prompt modulation
#         print("Check shape::::", out.shape, raw_flow.shape, motion_gate.shape)
        out = out * (1.0 + motion_gate)
        out_m = out_m * (1.0 + motion_gate)


        return out, out_m


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class IG_MSA(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)

    def forward(self, x):
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q_inp, k_inp, v_inp))
        q = F.normalize(q.transpose(-2, -1), dim=-1, p=2)
        k = F.normalize(k.transpose(-2, -1), dim=-1, p=2)
        v = v.transpose(-2, -1)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
        return self.proj(x).view(b, h, w, c)


class IG_MSA_memory(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, down_kernel=3, refine_act=nn.ReLU):
        super().__init__()
        self.num = 256
        self.momory_block = _Memory_Block_prompt(hdim=dim, kdim=self.num, moving_average_rate=0.9)
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.down = nn.Conv2d(dim, dim, kernel_size=down_kernel, stride=2, padding=down_kernel // 2, bias=False)
        self.up = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2, bias=False)
        self.refine = ResBlock(dim, dim, negative_slope=0.2, bias=False)
        self.training = True

    def dual_prompt_attention_lowres(self, x_lr_bhwc, z_lr_bhwc, z_m_lr_bhwc):
        b, h_lr, w_lr, c = x_lr_bhwc.shape

        q_h = self.to_q(x_lr_bhwc.reshape(b * h_lr, w_lr, c))
        k_h = self.to_k(z_lr_bhwc.reshape(b * h_lr, w_lr, c))
        v_h = self.to_v(z_m_lr_bhwc.reshape(b * h_lr, w_lr, c))
        q_h, k_h, v_h = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q_h, k_h, v_h))
        q_h = F.normalize(q_h, dim=-1)
        k_h = F.normalize(k_h, dim=-1)
        attn_h = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.rescale
        attn_h = attn_h.softmax(dim=-1)
        out_h = torch.matmul(attn_h, v_h)
        out_h = rearrange(out_h, "b h n d -> b n (h d)").reshape(b, h_lr, w_lr, c)

        q_v = self.to_q(x_lr_bhwc.permute(0, 2, 1, 3).reshape(b * w_lr, h_lr, c))
        k_v = self.to_k(z_lr_bhwc.permute(0, 2, 1, 3).reshape(b * w_lr, h_lr, c))
        v_v = self.to_v(z_m_lr_bhwc.permute(0, 2, 1, 3).reshape(b * w_lr, h_lr, c))
        q_v, k_v, v_v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q_v, k_v, v_v))
        q_v = F.normalize(q_v, dim=-1)
        k_v = F.normalize(k_v, dim=-1)
        attn_v = torch.matmul(q_v, k_v.transpose(-2, -1)) * self.rescale
        attn_v = attn_v.softmax(dim=-1)
        out_v = torch.matmul(attn_v, v_v)
        out_v = rearrange(out_v, "b h n d -> b n (h d)").reshape(b, w_lr, h_lr, c).permute(0, 2, 1, 3)

        out_lr = self.proj((out_h + out_v) * 0.5)
        return out_lr

    def forward(self, x, flow, origin_flow):
        z, z_m = self.momory_block(
            flow.permute(0, 3, 1, 2),
#             event.permute(0, 3, 1, 2),
            origin_flow,
            update_flag=self.training,
        )
        z = z.permute(0, 2, 3, 1)
        z_m = z_m.permute(0, 2, 3, 1)

        b, h, w, c = x.shape
        x_chw = x.permute(0, 3, 1, 2)
        x_lr = self.down(x_chw)
        h_lr, w_lr = x_lr.shape[2], x_lr.shape[3]
        z_lr = F.interpolate(z.permute(0, 3, 1, 2), size=(h_lr, w_lr), mode="bilinear", align_corners=False)
        z_m_lr = F.interpolate(z_m.permute(0, 3, 1, 2), size=(h_lr, w_lr), mode="bilinear", align_corners=False)

        x_lr_bhwc = x_lr.permute(0, 2, 3, 1)
        z_lr_bhwc = z_lr.permute(0, 2, 3, 1)
        z_m_lr_bhwc = z_m_lr.permute(0, 2, 3, 1)
        out_lr_bhwc = self.dual_prompt_attention_lowres(x_lr_bhwc, z_lr_bhwc, x_lr_bhwc)

        out_lr_chw = out_lr_bhwc.permute(0, 3, 1, 2)
        out_hr_chw = self.up(out_lr_chw)[:, :, :h, :w]
        out_chw = self.refine(out_hr_chw + x_chw)
        out = out_chw.permute(0, 2, 3, 1)
        return out


class UPT(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([IG_MSA_memory(dim=dim, dim_head=dim_head, heads=heads), PreNorm(dim, FeedForward(dim=dim))]))
            self.blocks.append(nn.ModuleList([IG_MSA(dim=dim, dim_head=dim_head, heads=heads), PreNorm(dim, FeedForward(dim=dim))]))

    def forward(self, x, flow, origin_flow):
        x = x.permute(0, 2, 3, 1)
        flow = flow.permute(0, 2, 3, 1)
#         event = event.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            if isinstance(attn, IG_MSA_memory):
                x = attn(x, flow, origin_flow) + x
            else:
                x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class ChannelLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(ChannelLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("Input must be 4D [B,C,H,W].")
        y = x.permute(0, 2, 3, 1).contiguous()
        y = self.norm(y)
        return y.permute(0, 3, 1, 2).contiguous()


class GEGLUFFN2D(nn.Module):
    def __init__(self, dim, expansion=2, bias=True):
        super(GEGLUFFN2D, self).__init__()
        hidden = dim * expansion
        self.proj_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.proj_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        u, g = self.proj_in(x).chunk(2, dim=1)
        y = self.proj_out(u * F.gelu(g))
        return y


def _pad_even_2d(x):
    h, w = x.shape[-2], x.shape[-1]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, pad_h, pad_w


def _haar_dwt2(x):
    x, pad_h, pad_w = _pad_even_2d(x)
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, lh, hl, hh, pad_h, pad_w


def _haar_iwt2(ll, lh, hl, hh, pad_h, pad_w):
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5

    bsz, ch, h, w = ll.shape
    out = torch.zeros((bsz, ch, h * 2, w * 2), device=ll.device, dtype=ll.dtype)
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d

    if pad_h:
        out = out[:, :, :-1, :]
    if pad_w:
        out = out[:, :, :, :-1]
    return out


def _haar_dwt1_channel(x):
    m = x.shape[1]
    pad = m % 2
    if pad:
        x = torch.cat([x, x[:, -1:, :, :]], dim=1)
    x0 = x[:, 0::2, :, :]
    x1 = x[:, 1::2, :, :]
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    low = (x0 + x1) * inv_sqrt2
    high = (x0 - x1) * inv_sqrt2
    return low, high, pad


def _haar_iwt1_channel(low, high, pad, out_m):
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    x0 = (low + high) * inv_sqrt2
    x1 = (low - high) * inv_sqrt2
    b, half, h, w = low.shape
    out = torch.zeros((b, half * 2, h, w), device=low.device, dtype=low.dtype)
    out[:, 0::2, :, :] = x0
    out[:, 1::2, :, :] = x1
    if pad:
        out = out[:, :out_m, :, :]
    return out


class WCFESpatialWaveletFilter(nn.Module):
    def __init__(self, dim: int, bias: bool = True):
        super(WCFESpatialWaveletFilter, self).__init__()
        self.g_ll = MixBlock(dim, out_channels=dim, bias=bias)
        self.g_lh = MixBlock(dim, out_channels=dim, bias=bias)
        self.g_hl = MixBlock(dim, out_channels=dim, bias=bias)
        self.g_hh = MixBlock(dim, out_channels=dim, bias=bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b_ll, b_lh, b_hl, b_hh, pad_h, pad_w = _haar_dwt2(x)
        c_ll, c_lh, c_hl, c_hh, _, _ = _haar_dwt2(y)
        
        f_ll = b_ll * self.g_ll(c_ll)
        f_lh = b_lh * self.g_lh(c_lh)
        f_hl = b_hl * self.g_hl(c_hl)
        f_hh = b_hh * self.g_hh(c_hh)

        return _haar_iwt2(f_ll, f_lh, f_hl, f_hh, pad_h, pad_w)


class WCFEWaveletChannelGate(nn.Module):
    def __init__(
        self,
        dim: int,
        bias: bool = True,
        reduction: int = 4,
    ):
        super(WCFEWaveletChannelGate, self).__init__()
        self.dim = dim
        self.m = dim

        half = (self.m + (self.m % 2)) // 2
        self.g_low = nn.Sequential(
            nn.Conv2d(half * 2, half, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, half, 1, bias=bias),
            nn.Sigmoid(),
        )
        self.high_refine = nn.Sequential(
            ChannelRefineBlock(feats=half),
            nn.Conv2d(half, half, 3, 1, 1, bias=bias),
        )
        self.g_high = MixBlock(half * 2, out_channels=half, hidden_channels=half, bias=bias)
        self.post_reduce = nn.Conv2d(self.m * 2, self.m, 1, 1, 0, bias=True)
        self.post_attn = SEBlock(self.m, reduction=reduction)
        self.post_refine = nn.Conv2d(self.m, self.m, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        guide_src = y + spatial
        b_low, b_high, pad = _haar_dwt1_channel(x)
        g_low, g_high, _ = _haar_dwt1_channel(guide_src)

        low_out = b_low * self.g_low(torch.cat([b_low, g_low], dim=1))

        high_feat = self.high_refine(b_high)
        high_gate = self.g_high(torch.cat([high_feat, g_high], dim=1))
        high_out = high_feat * high_gate

        gated_m = _haar_iwt1_channel(
            low_out,
            high_out,
            pad=pad,
            out_m=self.m,
        )
        out_m = self.post_reduce(torch.cat([gated_m, x], dim=1))
        out_m = out_m * self.post_attn(out_m)
        out_m = self.post_refine(out_m)
        return out_m


class WCFEBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion=2, bias=True):
        super(WCFEBlock, self).__init__()
        self.dim = dim

        self.norm_event = ChannelLayerNorm(dim)

        self.spatial_filter = WCFESpatialWaveletFilter(dim=dim, bias=bias)
        self.channel_gate = WCFEWaveletChannelGate(dim=dim, bias=bias)

        self.attn_norm_q = ChannelLayerNorm(dim)
        self.attn_norm_kv = ChannelLayerNorm(dim)
        self.cross_attn = Mutual_Attention(dim, num_heads, bias)

        self.ffn_norm = ChannelLayerNorm(dim)
        self.ffn = GEGLUFFN2D(dim=dim, expansion=ffn_expansion, bias=bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_n = self.norm_event(y)
        spatial = self.spatial_filter(x, y_n)
        channel = self.channel_gate(x, y_n, spatial)

        out = x + channel
        out = out + self.cross_attn(self.attn_norm_q(y), self.attn_norm_kv(x))
        out = out + self.ffn(self.ffn_norm(out))
        return out


__all__ = [
    "SAM",
    "UPT",
    "WCFEBlock",
    "WCFEWaveletChannelGate",
    "WCFESpatialWaveletFilter",
    "ChannelRefineBlock",
    "ChannelLayerNorm",
    "GEGLUFFN2D",
    "conv3x3",
    "scer_to_voxel_general",
    "_build_symmetric_ev_flow_feature",
]
