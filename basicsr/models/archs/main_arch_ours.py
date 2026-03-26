import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basicsr.models.archs.arch_util_ours import Mutual_Attention


def conv3x3(in_chn, out_chn, bias=True):
    return nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)


def conv_down(in_chn, out_chn, bias=False):
    return nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img

class ResidualBlock(nn.Module):
    def __init__(self, feats, distillation_rate=0.5):
        super(ResidualBlock, self).__init__()
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

class _Memory_Block_prompt(nn.Module):
    def __init__(self, hdim, kdim, moving_average_rate=0.999):
        super().__init__()
        self.c = hdim
        self.k = kdim
        self.moving_average_rate = moving_average_rate
        self.register_buffer("units", torch.randn(kdim, hdim))
        self.prompts = nn.Parameter(torch.rand(kdim, hdim), requires_grad=True)
        self.training = True

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

    def forward(self, x, update_flag=True):
        b, c, h, w = x.size()
        assert c == self.c
        x = x.permute(0, 2, 3, 1).reshape(-1, c)

        m = self.units
        p = self.prompts
        xn = F.normalize(x, dim=1)
        mn = F.normalize(m, dim=1)
        score = torch.matmul(xn, mn.t())

        if update_flag:
            m = self.update(x, score)
            mn = F.normalize(m, dim=1)
            score = torch.matmul(xn, mn.t())

        soft_label = F.softmax(score, dim=1)
        out_m = torch.matmul(soft_label, m).view(b, h, w, c).permute(0, 3, 1, 2)
        out = torch.matmul(soft_label, p).view(b, h, w, c).permute(0, 3, 1, 2)
        return out, out_m


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
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

    def forward(self, x, var):
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        var = var.reshape(b, h * w, c)
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
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            refine_act(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            refine_act(inplace=True),
        )
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

    def forward(self, x, var):
        z, z_m = self.momory_block(var.permute(0, 3, 1, 2), update_flag=self.training)
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
            self.blocks.append(
                nn.ModuleList(
                    [IG_MSA_memory(dim=dim, dim_head=dim_head, heads=heads), PreNorm(dim, FeedForward(dim=dim))]
                )
            )
            self.blocks.append(
                nn.ModuleList([IG_MSA(dim=dim, dim_head=dim_head, heads=heads), PreNorm(dim, FeedForward(dim=dim))])
            )

    def forward(self, x, var):
        x = x.permute(0, 2, 3, 1)
        var = var.permute(0, 2, 3, 1)
        scoremap = None
        for i, (attn, ff) in enumerate(self.blocks):
            if i == 0:
                x_ = attn(x, var)
                x = x_ + x
            else:
                x = attn(x, var) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out, scoremap


class EventGuidedVarFusion(nn.Module):
    def __init__(self, channels, max_disp=1):
        super().__init__()
        self.max_disp = max_disp
        hidden = max(channels // 4, 8)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def _cost_volume_flow(self, image_feat, event_feat):
        disps = []
        costs = []
        for dy in range(-self.max_disp, self.max_disp + 1):
            for dx in range(-self.max_disp, self.max_disp + 1):
                shifted = torch.roll(event_feat, shifts=(dy, dx), dims=(2, 3))
                cost = (image_feat * shifted).mean(dim=1, keepdim=True)
                costs.append(cost)
                disps.append((dx, dy))

        cost_volume = torch.cat(costs, dim=1)
        prob = F.softmax(cost_volume, dim=1)
        disp = image_feat.new_tensor(disps).view(1, -1, 2, 1, 1)
        flow_x = (prob * disp[:, :, 0]).sum(dim=1, keepdim=True)
        flow_y = (prob * disp[:, :, 1]).sum(dim=1, keepdim=True)
        return torch.cat([flow_x, flow_y], dim=1)

    def _warp(self, feat, flow):
        b, _, h, w = feat.shape
        yy, xx = torch.meshgrid(
            torch.arange(h, device=feat.device, dtype=feat.dtype),
            torch.arange(w, device=feat.device, dtype=feat.dtype),
        )
        base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        grid = base_grid + flow.permute(0, 2, 3, 1)
        grid_x = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
        grid_y = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
        norm_grid = torch.stack((grid_x, grid_y), dim=-1)
        return F.grid_sample(feat, norm_grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, image_feat, event_feat):
        event_feat = F.interpolate(event_feat, size=image_feat.shape[-2:], mode="bilinear", align_corners=False)
        flow = self._cost_volume_flow(image_feat, event_feat)
        warped_event = self._warp(event_feat, flow)
        attn = self.channel_attn(torch.cat([image_feat, warped_event], dim=1))
        return image_feat + attn * warped_event


class EventGuidedGlobalAttentionFusion(nn.Module):
    def __init__(self, channels, num_heads=4, kv_downsample=4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads for global_attn fusion")
        if kv_downsample < 1:
            raise ValueError("kv_downsample must be >= 1")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.kv_downsample = kv_downsample
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, image_feat, event_feat):
        b, c, h, w = image_feat.shape
        event_feat = F.interpolate(event_feat, size=(h, w), mode="bilinear", align_corners=False)
        if self.kv_downsample > 1:
            event_kv = F.avg_pool2d(event_feat, kernel_size=self.kv_downsample, stride=self.kv_downsample)
        else:
            event_kv = event_feat

        hk, wk = event_kv.shape[-2:]
        q = self.q_proj(image_feat).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        k = self.k_proj(event_kv).view(b, self.num_heads, self.head_dim, hk * wk).permute(0, 1, 3, 2)
        v = self.v_proj(event_kv).view(b, self.num_heads, self.head_dim, hk * wk).permute(0, 1, 3, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        out = self.out_proj(out)
        return image_feat + out


def _flatten_bt(x):
    if x.dim() == 4:
        b, c, h, w = x.shape
        return x, b, 1
    if x.dim() == 5:
        b, t, c, h, w = x.shape
        return x.reshape(b * t, c, h, w), b, t
    raise ValueError("Input must be 4D [B,C,H,W] or 5D [B,T,C,H,W].")


def _unflatten_bt(x, b, t):
    if t == 1:
        return x
    bt, c, h, w = x.shape
    if bt != b * t:
        raise ValueError("Invalid [B,T] for unflatten.")
    return x.view(b, t, c, h, w)


def _merge_tc(x):
    if x.dim() == 4:
        return x
    b, t, c, h, w = x.shape
    return x.view(b, t * c, h, w)


def _split_tc(x, t, c):
    if t == 1:
        return x
    b, m, h, w = x.shape
    if m != t * c:
        raise ValueError("Channel size mismatch while splitting (T,C).")
    return x.view(b, t, c, h, w)


class ChannelLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(ChannelLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        if x.dim() == 4:
            y = x.permute(0, 2, 3, 1).contiguous()
            y = self.norm(y)
            return y.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 5:
            y = x.permute(0, 1, 3, 4, 2).contiguous()
            y = self.norm(y)
            return y.permute(0, 1, 4, 2, 3).contiguous()
        raise ValueError("Input must be 4D or 5D.")


class GEGLUFFN2D(nn.Module):
    def __init__(self, dim, expansion=2, bias=True):
        super(GEGLUFFN2D, self).__init__()
        hidden = dim * expansion
        self.proj_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.proj_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x4, b, t = _flatten_bt(x)
        u, g = self.proj_in(x4).chunk(2, dim=1)
        y = self.proj_out(u * F.gelu(g))
        return _unflatten_bt(y, b, t)


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
    def __init__(self, dim, bias=True):
        super(WCFESpatialWaveletFilter, self).__init__()
        self.g_ll = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=bias), nn.Sigmoid())
        self.g_lh = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=bias), nn.Sigmoid())
        self.g_hl = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=bias), nn.Sigmoid())
        self.g_hh = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=bias), nn.Sigmoid())

    def forward(self, blur_4d, corr_4d):
        b_ll, b_lh, b_hl, b_hh, pad_h, pad_w = _haar_dwt2(blur_4d)
        c_ll, c_lh, c_hl, c_hh, _, _ = _haar_dwt2(corr_4d)
        f_ll = b_ll * self.g_ll(c_ll)
        f_lh = b_lh * self.g_lh(c_lh)
        f_hl = b_hl * self.g_hl(c_hl)
        f_hh = b_hh * self.g_hh(c_hh)
        return _haar_iwt2(f_ll, f_lh, f_hl, f_hh, pad_h, pad_w)


class WCFEChannelGate(nn.Module):
    def __init__(self, dim, num_frames=1, bias=True):
        super(WCFEChannelGate, self).__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.m = dim * num_frames
        half = (self.m + (self.m % 2)) // 2
        self.g_low = nn.Sequential(nn.Conv2d(half, half, 1, bias=bias), nn.Sigmoid())
        self.g_high = nn.Sequential(nn.Conv2d(half, half, 1, bias=bias), nn.Sigmoid())

    def forward(self, blur, corr, spatial):
        del spatial
        blur_m = _merge_tc(blur)
        mod_src = _merge_tc(corr)
        if blur_m.shape[1] != self.m:
            raise ValueError("WCFEChannelGate expects fixed T and C (M=T*C) from init.")

        b_low, b_high, pad = _haar_dwt1_channel(blur_m)
        s_low, s_high, _ = _haar_dwt1_channel(mod_src)
        out_m = _haar_iwt1_channel(
            b_low * self.g_low(s_low),
            b_high * self.g_high(s_high),
            pad=pad,
            out_m=self.m,
        )
        return _split_tc(out_m, self.num_frames, self.dim)


class WCFEBlock(nn.Module):
    def __init__(self, dim, num_frames=1, num_heads=4, ffn_expansion=2, bias=True):
        super(WCFEBlock, self).__init__()
        self.dim = dim
        self.num_frames = num_frames

        self.norm_blur = ChannelLayerNorm(dim)
        self.norm_event = ChannelLayerNorm(dim)
        self.corr = nn.Conv2d(dim * 2, dim, 1, bias=bias)

        self.spatial_filter = WCFESpatialWaveletFilter(dim=dim, bias=bias)
        self.channel_gate = WCFEChannelGate(dim=dim, num_frames=num_frames, bias=bias)

        self.attn_norm_q = ChannelLayerNorm(dim)
        self.attn_norm_kv = ChannelLayerNorm(dim)
        self.cross_attn = Mutual_Attention(dim, num_heads, bias)

        self.ffn_norm = ChannelLayerNorm(dim)
        self.ffn = GEGLUFFN2D(dim=dim, expansion=ffn_expansion, bias=bias)

    def forward(self, blur_feat, event_feat):
        if blur_feat.shape != event_feat.shape:
            raise ValueError("blur_feat and event_feat must have the same shape.")
        if blur_feat.dim() == 5 and blur_feat.shape[1] != self.num_frames:
            raise ValueError("Input T must match WCFEBlock(num_frames=...).")
        if blur_feat.dim() not in (4, 5):
            raise ValueError("Input must be 4D or 5D.")

        blur_n = self.norm_blur(blur_feat)
        event_n = self.norm_event(event_feat)
        blur4, b, t = _flatten_bt(blur_n)
        event4, _, _ = _flatten_bt(event_n)

        corr4 = self.corr(torch.cat([blur4, event4], dim=1))
        spatial4 = self.spatial_filter(_flatten_bt(blur_feat)[0], corr4)

        corr = _unflatten_bt(corr4, b, t)
        spatial = _unflatten_bt(spatial4, b, t)
        channel = self.channel_gate(blur_feat, corr, spatial)
        attn_q = self.attn_norm_q(blur_feat)
        attn_kv = self.attn_norm_kv(event_feat)
        attn = _unflatten_bt(self.cross_attn(_flatten_bt(attn_q)[0], _flatten_bt(attn_kv)[0]), b, t)
        x = blur_feat + spatial + channel
        x = x + attn
        x = x + self.ffn(self.ffn_norm(x))
        return x


__all__ = [
    "SAM",
    "UPT",
    "EventGuidedVarFusion",
    "EventGuidedGlobalAttentionFusion",
    "WCFEBlock",
    "WCFEChannelGate",
    "WCFESpatialWaveletFilter",
    "ChannelLayerNorm",
    "GEGLUFFN2D",
    "conv3x3",
    "ResidualBlock"
]
