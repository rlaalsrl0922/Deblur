import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


def _fft2_ri(x, norm="backward"):
    """Return 2D FFT real/imag concatenation along channel dim."""
    fft = torch.fft.fft2(x, norm=norm)
    return torch.cat([fft.real, fft.imag], dim=1)


def _fft2_mag(x, norm="backward"):
    """Return 2D FFT magnitude."""
    fft = torch.fft.fft2(x, norm=norm)
    return torch.abs(fft)


def _haar_dwt2_cat(x):
    """Return 2D Haar DWT subbands concatenated on channel dim: [LL,LH,HL,HH]."""
    b, c, h, w = x.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    filt = x.new_tensor([
        [[1.0, 1.0], [1.0, 1.0]],
        [[-1.0, -1.0], [1.0, 1.0]],
        [[-1.0, 1.0], [-1.0, 1.0]],
        [[1.0, -1.0], [-1.0, 1.0]],
    ]) * 0.5
    weight = filt.view(4, 1, 2, 2).repeat(c, 1, 1, 1)  # (4c,1,2,2)
    y = F.conv2d(x, weight, stride=2, padding=0, groups=c)  # (b,4c,h/2,w/2)
    return y


def _haar_dwt2_subbands(x):
    """Return one-level 2D Haar DWT subbands as tuple: (LL, LH, HL, HH)."""
    y = _haar_dwt2_cat(x)
    b, c4, h2, w2 = y.shape
    c = c4 // 4
    y = y.view(b, c, 4, h2, w2)
    return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# AT loss
def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()

@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)
    
# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class SRNLoss(nn.Module):

    def __init__(self):
        super(SRNLoss, self).__init__()  

    def forward(self, preds, target):

        gt1 = target
        B,C,H,W = gt1.shape
        gt2 = F.interpolate(gt1, size=(H // 2, W // 2), mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, size=(H // 4, W // 4), mode='bilinear', align_corners=False)

        l1 = mse_loss(preds[0] , gt3)
        l2 = mse_loss(preds[1] , gt2)
        l3 = mse_loss(preds[2] , gt1)

        return l1+l2+l3



class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).
    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".
    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


class DualDomainL1Loss(nn.Module):
    """Dual-domain L1 loss (spatial + frequency L1).

    L = ||pred - target||_1 + lambda * ||F(pred) - F(target)||_1
    """

    def __init__(
        self,
        loss_weight=1.0,
        reduction='mean',
        freq_weight=0.1,
        fft_norm='backward',
        transform='fft',
        dwt_levels=1,
        dwt_band_weights=None,
        dwt_level_weights=None,
    ):
        super(DualDomainL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        if transform not in ['fft', 'dwt']:
            raise ValueError(f'Unsupported transform: {transform}. Supported ones are: ["fft", "dwt"]')
        if dwt_levels < 1:
            raise ValueError('dwt_levels must be >= 1.')
        if transform == 'dwt' and reduction == 'none':
            raise ValueError('DualDomainL1Loss with transform="dwt" does not support reduction="none".')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.freq_weight = freq_weight
        self.fft_norm = fft_norm
        self.transform = transform
        self.dwt_levels = int(dwt_levels)
        self.dwt_band_weights = self._parse_band_weights(dwt_band_weights)
        self.dwt_level_weights = self._parse_level_weights(dwt_level_weights, self.dwt_levels)

    @staticmethod
    def _parse_band_weights(dwt_band_weights):
        default_weights = {'LL': 1.0, 'LH': 1.0, 'HL': 1.0, 'HH': 1.0}
        if dwt_band_weights is None:
            return default_weights
        if isinstance(dwt_band_weights, dict):
            for band in default_weights:
                if band in dwt_band_weights:
                    default_weights[band] = float(dwt_band_weights[band])
            return default_weights
        if isinstance(dwt_band_weights, (list, tuple)):
            if len(dwt_band_weights) != 4:
                raise ValueError('dwt_band_weights must have exactly 4 values: [LL, LH, HL, HH].')
            return {
                'LL': float(dwt_band_weights[0]),
                'LH': float(dwt_band_weights[1]),
                'HL': float(dwt_band_weights[2]),
                'HH': float(dwt_band_weights[3]),
            }
        raise ValueError('dwt_band_weights must be dict or list/tuple.')

    @staticmethod
    def _parse_level_weights(dwt_level_weights, dwt_levels):
        if dwt_level_weights is None:
            return [1.0] * dwt_levels
        if isinstance(dwt_level_weights, (int, float)):
            return [float(dwt_level_weights)] * dwt_levels
        if isinstance(dwt_level_weights, dict):
            level_weights = []
            for lv in range(dwt_levels):
                key = f'lv{lv + 1}'
                level_weights.append(float(dwt_level_weights.get(key, 1.0)))
            return level_weights
        if isinstance(dwt_level_weights, (list, tuple)):
            if len(dwt_level_weights) != dwt_levels:
                raise ValueError(f'dwt_level_weights length ({len(dwt_level_weights)}) must match dwt_levels ({dwt_levels}).')
            return [float(v) for v in dwt_level_weights]
        raise ValueError('dwt_level_weights must be scalar, dict, or list/tuple.')

    def _dwt_subband_loss(self, pred, target):
        cur_pred, cur_target = pred, target
        loss = 0.0
        for lv in range(self.dwt_levels):
            pred_ll, pred_lh, pred_hl, pred_hh = _haar_dwt2_subbands(cur_pred)
            tgt_ll, tgt_lh, tgt_hl, tgt_hh = _haar_dwt2_subbands(cur_target)

            lv_weight = self.dwt_level_weights[lv]
            band_loss = (
                self.dwt_band_weights['LL'] * l1_loss(pred_ll, tgt_ll, reduction=self.reduction) +
                self.dwt_band_weights['LH'] * l1_loss(pred_lh, tgt_lh, reduction=self.reduction) +
                self.dwt_band_weights['HL'] * l1_loss(pred_hl, tgt_hl, reduction=self.reduction) +
                self.dwt_band_weights['HH'] * l1_loss(pred_hh, tgt_hh, reduction=self.reduction)
            )
            loss = loss + lv_weight * band_loss
            cur_pred, cur_target = pred_ll, tgt_ll
        return loss

    def forward(self, pred, target, weight=None, **kwargs):
        l_spatial = l1_loss(pred, target, weight, reduction=self.reduction)
        if self.transform == 'fft':
            pred_t = _fft2_ri(pred, norm=self.fft_norm)
            target_t = _fft2_ri(target, norm=self.fft_norm)
            l_freq = l1_loss(pred_t, target_t, reduction=self.reduction)
        else:  # dwt
            l_freq = self._dwt_subband_loss(pred, target)
        return self.loss_weight * (l_spatial + self.freq_weight * l_freq)


class CharbonnierFRLoss(nn.Module):
    """Charbonnier (spatial) + frequency reconstruction L1 loss."""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12, freq_weight=0.01, fft_norm='backward'):
        super(CharbonnierFRLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps
        self.freq_weight = freq_weight
        self.fft_norm = fft_norm

    def forward(self, pred, target, weight=None, **kwargs):
        l_spatial = charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)
        pred_f = _fft2_ri(pred, norm=self.fft_norm)
        target_f = _fft2_ri(target, norm=self.fft_norm)
        l_freq = l1_loss(pred_f, target_f, reduction=self.reduction)
        return self.loss_weight * (l_spatial + self.freq_weight * l_freq)


class FrequencyL1Loss(nn.Module):
    """Frequency-domain L1 loss on FFT real/imag."""

    def __init__(self, loss_weight=1.0, reduction='mean', fft_norm='backward'):
        super(FrequencyL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.fft_norm = fft_norm

    def forward(self, pred, target, weight=None, **kwargs):
        pred_f = _fft2_ri(pred, norm=self.fft_norm)
        target_f = _fft2_ri(target, norm=self.fft_norm)
        l_freq = l1_loss(pred_f, target_f, reduction=self.reduction)
        return self.loss_weight * l_freq


class FrequencyMagnitudeL1Loss(nn.Module):
    """Frequency-domain L1 loss on FFT magnitude."""

    def __init__(self, loss_weight=1.0, reduction='mean', fft_norm='backward'):
        super(FrequencyMagnitudeL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.fft_norm = fft_norm

    def forward(self, pred, target, weight=None, **kwargs):
        pred_f = _fft2_mag(pred, norm=self.fft_norm)
        target_f = _fft2_mag(target, norm=self.fft_norm)
        l_freq = l1_loss(pred_f, target_f, reduction=self.reduction)
        return self.loss_weight * l_freq

class WeightedTVLoss(L1Loss):
    """Weighted TV loss.
        Args:
            loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super(WeightedTVLoss, self).forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super(WeightedTVLoss, self).forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss
