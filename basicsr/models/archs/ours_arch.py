import torch
import torch.nn as nn

from basicsr.models.archs.main_arch_ours import (
    ChannelRefineBlock,
    UPT,
    WCFEBlock,
    _build_symmetric_ev_flow_feature,
    scer_to_voxel_general,
)
from basicsr.models.archs.arch_util_ours import (
    EventImage_ChannelAttentionTransformerBlock,
    ResBlock,
    SAM,
    conv_down,
)

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
        var_fusion_max_disp=3,
        var_fusion_num_heads=4,
        var_fusion_inner_radius=0,
    ):
        super(EFNetOurs, self).__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        self.var_fusion_max_disp = var_fusion_max_disp
        self.var_fusion_num_heads = var_fusion_num_heads
        self.var_fusion_inner_radius = var_fusion_inner_radius
        self.down_path_1 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        
        self.down_path_ev = nn.ModuleList()
        self.conv_ev1 = nn.Conv2d(ev_chn, wf, 3, 1, 1)
        
        self.down_path_ev_flow = nn.ModuleList()
        self.ev_flow_depth = max(self.depth - 1, 0)
        self.ev_flow_downsample = 2 ** max(self.ev_flow_depth - 1, 0)
        self.ev_flow_channels = (ev_chn-2) * 2
        self.conv_ev_flow1 = nn.Conv2d(self.ev_flow_channels, wf, 3, 1, 1)
        self.ev_flow_stem = ResBlock(self.ev_flow_channels, self.ev_flow_channels, negative_slope=0.2)

        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            downsample = True if (i+1) < depth else False 
            ev_flow_downsample = True if (i+1) < self.ev_flow_depth else False
            stage_channels = wf

            self.down_path_1.append(
                UNetConvBlock(
                    prev_channels,
                    stage_channels,
                    downsample,
                    relu_slope,
                    num_heads=self.num_heads[i],
                    use_wcfe=self.num_heads[i],
                )
            )
            # ev encoder
            self.down_path_ev.append(UNetEVConvBlock(prev_channels, stage_channels, downsample , relu_slope))
            if i < self.ev_flow_depth:
                self.down_path_ev_flow.append(UNetEVConvBlock(prev_channels, stage_channels, ev_flow_downsample, relu_slope,))

            prev_channels = stage_channels

        self.up_path_1 = nn.ModuleList()
        self.up_path_1_memory = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.var_flow_stage1 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            stage_channels = wf
            self.up_path_1.append(
                UNetUpBlock(
                    prev_channels,
                    wf,
                    relu_slope,
                    num_heads=self.num_heads[i],
                    use_wcfe=self.num_heads[i],
                )
            )
            self.up_path_1_memory.append(UPT(dim=stage_channels, num_blocks=1, dim_head=stage_channels, heads=1))
            self.skip_conv_1.append(nn.Conv2d(wf, wf, 3, 1, 1))
            self.var_flow_stage1.append(nn.Sequential(
                nn.Conv2d(stage_channels, stage_channels, 3, 1, 1),
                nn.ELU(),
                nn.Conv2d(stage_channels, stage_channels, 3, 1, 1),
                nn.ELU(),
            ))
            prev_channels = wf
            
        self.sam12 = SAM(prev_channels)

        self.last_var_head = nn.Sequential(
            nn.Conv2d(prev_channels, prev_channels, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(prev_channels, in_chn, 1, 1, 0),
        )

    def forward(self, x, event, mask=None):
        image = x

        # Event preprocessing follows the downstream [B, 8, H, W] convention.
        event = torch.cat((event[:, :4, :, :], event[:, 5:, :, :]), dim=1)
        event = scer_to_voxel_general(event)
        max_abs = event.abs().max()
        if max_abs > 0:
            event = event/ max_abs 

        ev_flow = []
        flow_feat, flow = _build_symmetric_ev_flow_feature(event,self.ev_flow_stem,self.ev_flow_downsample,radius=self.var_fusion_max_disp,inner_radius=self.var_fusion_inner_radius,temperature=0.1)
        flow_feat = self.conv_ev_flow1(flow_feat)
        for i, down in enumerate(self.down_path_ev_flow):
            if i < self.ev_flow_depth - 1:
                flow_feat, flow_feat_up = down(flow_feat, self.fuse_before_downsample)
                ev_flow.append(flow_feat_up if self.fuse_before_downsample else flow_feat)
            else:
                flow_feat = down(flow_feat, self.fuse_before_downsample)
                ev_flow.append(flow_feat)

        ev = []
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
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:

                x1, x1_up = down(x1, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)
                encs.append(x1_up)
            
            else:
                x1 = down(x1, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)

        for i, up in enumerate(self.up_path_1):
            ev_idx = self.depth - i - 2
            x1 = up(x1, self.skip_conv_1[i](encs[-i-1]), event_filter=ev[ev_idx])
            
            var_ev_flow = self.var_flow_stage1[i](ev_flow[ev_idx])
            
            x1 = self.up_path_1_memory[i](x1, var_ev_flow, flow)

        var_feature = self.last_var_head(x1)
        _, out_1 = self.sam12(x1, image)
        return [out_1], [var_feature]

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
        num_heads=None,
        use_wcfe=None,
    ):
        super(UNetConvBlock, self).__init__()
        self.downsample = conv_down(out_size, out_size, bias=False) if downsample else None
        self.num_heads = num_heads
        self.use_wcfe = use_wcfe

        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.res_block = ResBlock(in_size, out_size, negative_slope=relu_slope)
        self.slb = ChannelRefineBlock(feats=out_size)

        if self.num_heads is not None:
            self.image_event_transformer = EventImage_ChannelAttentionTransformerBlock(
                out_size,
                num_heads=self.num_heads,
                ffn_expansion_factor=4,
                bias=False,
                LayerNorm_type="WithBias",
            )

        if self.use_wcfe:
            self.image_event_wcfe = WCFEBlock(
                dim=out_size,
                num_heads=self.num_heads,
                ffn_expansion=4,
                bias=False,
            )

    def forward(self, x, enc=None, event_filter=None, merge_before_downsample=True):
        out = self.res_block(x) + self.identity(x)
        out = self.slb(out) + out

        if self.num_heads is not None:
            out = self.image_event_transformer(out, event_filter)
        if self.use_wcfe:
            out = self.image_event_wcfe(out, event_filter)

        if self.downsample is not None:
            out_down = self.downsample(out)
            if not merge_before_downsample:
                out_down = self.image_event_transformer(out_down, event_filter)
                if self.use_wcfe:
                    out = self.image_event_wcfe(out, event_filter)
            return out_down, out

        if merge_before_downsample:
            return out
        out = self.image_event_transformer(out, event_filter)
        if self.use_wcfe:
            out = self.image_event_wcfe(out, event_filter)


class UNetEVConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope):
        super(UNetEVConvBlock, self).__init__()
        self.downsample = conv_down(out_size, out_size, bias=False) if downsample else None
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.res_block = ResBlock(in_size, out_size, negative_slope=relu_slope)
        self.conv_before_merge = nn.Conv2d(out_size, out_size, 1, 1, 0)

    def forward(self, x, merge_before_downsample=True):
        out = self.res_block(x) + self.identity(x)

        if self.downsample is not None:
            out_down = self.downsample(out)
            if not merge_before_downsample:
                out_down = self.conv_before_merge(out_down)
            else:
                out = self.conv_before_merge(out)
            return out_down, out

        out = self.conv_before_merge(out)
        return out


class UNetUpBlock(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        relu_slope,
        num_heads=None,
        use_wcfe=None,
    ):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(
            out_size * 2,
            out_size,
            False,
            relu_slope,
            num_heads=num_heads,
            use_wcfe=use_wcfe,
        )

    def forward(self, x, bridge, event_filter=None):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out, event_filter=event_filter)
        return out
