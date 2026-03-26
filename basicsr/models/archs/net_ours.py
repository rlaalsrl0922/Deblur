import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.main_arch_ours import (
    UPT,
    EventGuidedGlobalAttentionFusion,
    EventGuidedVarFusion,
    WCFEBlock,
    ResidualBlock
)
from basicsr.models.archs.arch_util_ours import EventImage_ChannelAttentionTransformerBlock_w_DCN,EventImage_ChannelAttentionTransformerBlock, ResidualBlock, conv_down, conv3x3, SAM

class EFNetOurs(nn.Module):
    def __init__(
        self,
        in_chn=3,
        ev_chn=8,
        wf=64,
        depth=3,
        fuse_before_downsample=True,
        relu_slope=0.2,
        num_heads=[1, 2, 4],
        var_fusion_type="cvwarp",
        var_fusion_max_disp=1,
        var_fusion_num_heads=4,
        var_fusion_kv_downsample=4,
    ):
        super(EFNetOurs, self).__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        if var_fusion_type not in {"cvwarp", "global_attn"}:
            raise ValueError("var_fusion_type must be one of: {'cvwarp', 'global_attn'}")
        self.var_fusion_type = var_fusion_type
        self.var_fusion_max_disp = var_fusion_max_disp
        self.var_fusion_num_heads = var_fusion_num_heads
        self.var_fusion_kv_downsample = var_fusion_kv_downsample
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.down_path_ev = nn.ModuleList()
        self.conv_ev1 = nn.Conv2d(ev_chn, wf, 3, 1, 1)

        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            downsample = True if (i + 1) < depth else False

            self.down_path_1.append(
                UNetConvBlock(
                    prev_channels,
                    (2**i) * wf,
                    downsample,
                    relu_slope,
                    num_heads=self.num_heads[i],
                    use_wcfe=self.num_heads[i],
                    use_hf_branch=False,
                )
            )
            self.down_path_2.append(
                UNetConvBlock(
                    prev_channels,
                    (2**i) * wf,
                    downsample,
                    relu_slope,
                    use_hf_branch=False,
                )
            )
            self.down_path_ev.append(UNetEVConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope))
            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.up_path_1_memory = nn.ModuleList()
        self.up_path_1_var_fusion = nn.ModuleList()
        self.up_path_1_var = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            stage_channels = (2**i) * wf
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i) * wf, relu_slope))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2**i) * wf, relu_slope))
            self.up_path_1_memory.append(UPT(dim=stage_channels, num_blocks=1, dim_head=stage_channels, heads=1))
            if self.var_fusion_type == "cvwarp":
                self.up_path_1_var_fusion.append(
                    EventGuidedVarFusion(channels=stage_channels, max_disp=self.var_fusion_max_disp)
                )
            else:
                self.up_path_1_var_fusion.append(
                    EventGuidedGlobalAttentionFusion(
                        channels=stage_channels,
                        num_heads=self.var_fusion_num_heads,
                        kv_downsample=self.var_fusion_kv_downsample,
                    )
                )
            self.up_path_1_var.append(
                nn.Sequential(
                    nn.Conv2d(stage_channels, stage_channels, 3, 1, 1),
                    nn.ELU(),
                    nn.Conv2d(stage_channels, stage_channels, 3, 1, 1),
                    nn.ELU(),
                )
            )
            self.skip_conv_1.append(nn.Conv2d((2**i) * wf, (2**i) * wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2**i) * wf, (2**i) * wf, 3, 1, 1))
            prev_channels = (2**i) * wf
        self.sam12 = SAM(prev_channels)

        self.cat12 = nn.Conv2d(prev_channels * 2, prev_channels, 1, 1, 0)
        self.last = conv3x3(prev_channels, in_chn, bias=True)
        self.var_head_stage1 = nn.Sequential(
            nn.Conv2d(prev_channels, prev_channels, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(prev_channels, prev_channels, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(prev_channels, in_chn, 1, 1, 0),
        )

    def forward(self, x, event, mask=None):
        del mask
        image = x

        ev = []
        event = torch.cat((event[:, :4, :, :], event[:, 5:, :, :]), dim=1)
        e1 = self.conv_ev1(event)
        for i, down in enumerate(self.down_path_ev):
            if i < self.depth - 1:
                e1, e1_up = down(e1, self.fuse_before_downsample)
                ev.append(e1_up if self.fuse_before_downsample else e1)
            else:
                e1 = down(e1, self.fuse_before_downsample)
                ev.append(e1)

        x1 = self.conv_01(image)
        encs = []
        decs = []
        last_stage1_var_feat = None
        for i, down in enumerate(self.down_path_1):
            if (i + 1) < self.depth:
                x1, x1_up = down(x1, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)
                encs.append(x1_up)
            else:
                x1 = down(x1, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)

        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i - 1]))
            ev_idx = self.depth - i - 2
            fused_event_feat = self.up_path_1_var_fusion[i](x1, ev[ev_idx])
            var_feat = self.up_path_1_var[i](fused_event_feat)
            x1, _ = self.up_path_1_memory[i](x1, var_feat)
            last_stage1_var_feat = var_feat
            decs.append(x1)
        if last_stage1_var_feat is None:
            raise RuntimeError("Stage 1 var feature was not generated.")
        var_stage1 = self.var_head_stage1(last_stage1_var_feat)
        sam_feature, out_1 = self.sam12(x1, image)

        x2 = self.conv_02(image)
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if (i + 1) < self.depth:
                x2, x2_up = down(
                    x2,
                    encs[i],
                    decs[-i - 1],
                    event_filter=ev[i],
                    merge_before_downsample=self.fuse_before_downsample,
                )
                blocks.append(x2_up)
            else:
                x2 = down(
                    x2,
                    event_filter=ev[i],
                    merge_before_downsample=self.fuse_before_downsample,
                )

        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i - 1]))

        out_2 = self.last(x2)
        out_2 = out_2 + image
        return [out_1, out_2], [var_stage1]

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain("leaky_relu", 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class UNetConvBlock(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        downsample,
        relu_slope,
        use_emgc=False,
        num_heads=None,
        use_wcfe=None,
        use_hf_branch=False,
    ):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc
        self.num_heads = num_heads
        self.use_wcfe = use_wcfe
        self.use_hf_branch = use_hf_branch

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.slb = ResidualBlock(feats=out_size)

        if self.use_hf_branch:
            self.hf_conv = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
            self.hf_event_gate = nn.Conv2d(out_size * 2, out_size, kernel_size=1, padding=0, bias=True)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        if self.num_heads is not None:
            self.image_event_transformer = EventImage_ChannelAttentionTransformerBlock_w_DCN(
                out_size,
                num_heads=self.num_heads,
                ffn_expansion_factor=4,
                bias=False,
                LayerNorm_type="WithBias",
            )

        if self.use_wcfe is not None:
            self.image_event_wcfe = WCFEBlock(
                dim=out_size,
                num_frames=1,
                num_heads=self.use_wcfe,
                ffn_expansion=4,
                bias=False,
            )

    def forward(self, x, enc=None, dec=None, mask=None, event_filter=None, merge_before_downsample=True):
        out = self.conv_1(x)
        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))
        out = out_conv2 + self.identity(x)
        out = self.slb(out) + out

        if self.use_hf_branch:
            hf = out - F.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
            hf = self.hf_conv(hf)
            gate = torch.sigmoid(self.hf_event_gate(torch.cat([out, event_filter], dim=1)))
            out = out + gate * hf

        if enc is not None and dec is not None and mask is not None:
            assert self.use_emgc
            out_enc = self.emgc_enc(enc) + self.emgc_enc_mask((1 - mask) * enc)
            out_dec = self.emgc_dec(dec) + self.emgc_dec_mask(mask * dec)
            out = out + out_enc + out_dec

        if self.num_heads is not None:
            out = self.image_event_transformer(out, event_filter)
        if self.use_wcfe is not None:
            out = self.image_event_wcfe(out, event_filter)

        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample:
                out_down = self.image_event_transformer(out_down, event_filter)
                if self.use_wcfe is not None:
                    out = self.image_event_wcfe(out, event_filter)
            return out_down, out

        if merge_before_downsample:
            return out
        out = self.image_event_transformer(out, event_filter)
        if self.use_wcfe is not None:
            out = self.image_event_wcfe(out, event_filter)


class UNetEVConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False):
        super(UNetEVConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_before_merge = nn.Conv2d(out_size, out_size, 1, 1, 0)

        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, merge_before_downsample=True):
        out = self.conv_1(x)
        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))
        out = out_conv2 + self.identity(x)

        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample:
                out_down = self.conv_before_merge(out_down)
            else:
                out = self.conv_before_merge(out)
            return out_down, out

        out = self.conv_before_merge(out)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out
