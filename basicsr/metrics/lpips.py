import numpy as np
import torch
import pyiqa

from basicsr.metrics.metric_util import reorder_image, to_y_channel


_LPIPS_METRIC_CACHE = {}


def _get_lpips_metric(device):
    """Create or reuse cached LPIPS metric."""
    device = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    key = str(device)

    if key not in _LPIPS_METRIC_CACHE:
        _LPIPS_METRIC_CACHE[key] = pyiqa.create_metric('lpips', device=device)
    return _LPIPS_METRIC_CACHE[key], device


def _to_numpy_image(img, input_order='HWC'):
    """Convert torch.Tensor / np.ndarray to numpy image in HWC order."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.dim() == 4:
            if img.size(0) != 1:
                raise ValueError(f'Only batch size 1 is supported, but got shape {img.shape}')
            img = img.squeeze(0)
        if img.dim() == 3:
            img = img.numpy().transpose(1, 2, 0)  # CHW -> HWC
        elif img.dim() == 2:
            img = img.numpy()[..., None]  # HW -> HWC with C=1
        else:
            raise ValueError(f'Unsupported tensor shape: {img.shape}')
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = img[..., None]  # HW -> HWC with C=1
    else:
        raise TypeError(f'img should be np.ndarray or torch.Tensor, but got {type(img)}')

    img = reorder_image(img, input_order=input_order)
    return img


def _normalize_to_01(img):
    """Normalize image to [0, 1]."""
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)
    return img


def _prepare_lpips_tensor(img, device):
    """
    Convert HWC numpy image in [0,1] to torch tensor (1,3,H,W) on device.
    LPIPS expects RGB, 0~1.
    """
    if img.ndim != 3:
        raise ValueError(f'Expected 3D HWC image, but got shape {img.shape}')

    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] != 3:
        raise ValueError(f'LPIPS expects 1 or 3 channel image, but got shape {img.shape}')

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def calculate_lpips(img1,
                    img2,
                    crop_border,
                    input_order='HWC',
                    test_y_channel=False,
                    device=None):
    """Calculate LPIPS.

    Args:
        img1 (ndarray/tensor): Images with range [0, 255] or [0, 1].
        img2 (ndarray/tensor): Images with range [0, 255] or [0, 1].
        crop_border (int): Cropped pixels in each edge of an image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): If True, convert to Y channel first.
            Since LPIPS is defined for RGB-like inputs, Y channel is replicated
            to 3 channels internally.
        device (str | torch.device | None): Device for LPIPS metric.

    Returns:
        float: lpips result. Lower is better.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        )

    img1 = _to_numpy_image(img1, input_order=input_order)
    img2 = _to_numpy_image(img2, input_order=input_order)

    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.'
    )

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
        # to_y_channel outputs HWC with C=1, later replicated to 3ch

    img1 = _normalize_to_01(img1)
    img2 = _normalize_to_01(img2)

    metric, device = _get_lpips_metric(device)

    img1_tensor = _prepare_lpips_tensor(img1, device)
    img2_tensor = _prepare_lpips_tensor(img2, device)

    with torch.no_grad():
        score = metric(img1_tensor, img2_tensor)

    if isinstance(score, torch.Tensor):
        score = score.item()

    return float(score)