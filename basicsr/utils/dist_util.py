# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py  # noqa: E501
import functools
import os
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta


def _get_process_group_timeout():
    """Return the c10d timeout, with a per-job environment override.

    PyTorch treats ``timeout=None`` as "use its default" rather than as an
    infinite timeout.  Keep the previous 60-minute default for other jobs,
    while allowing long-running jobs to request an effectively unlimited
    timeout explicitly.
    """
    timeout_days = os.environ.get('TORCH_DIST_TIMEOUT_DAYS')
    if timeout_days is None:
        return timedelta(minutes=60)

    timeout_days = int(timeout_days)
    if timeout_days <= 0:
        raise ValueError('TORCH_DIST_TIMEOUT_DAYS must be positive')
    return timedelta(days=timeout_days)


def _get_rendezvous_timeout():
    """Keep startup/peer discovery failures on a normal timeout."""
    timeout_seconds = int(
        os.environ.get('TORCH_DIST_RENDEZVOUS_TIMEOUT_SECONDS', '300'))
    if timeout_seconds <= 0:
        raise ValueError(
            'TORCH_DIST_RENDEZVOUS_TIMEOUT_SECONDS must be positive')
    return timedelta(seconds=timeout_seconds)


def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)

    # Keep the rendezvous socket timeout independent from the collective
    # timeout.  Otherwise a deliberately long NCCL collective timeout also
    # makes a missing peer look like a multi-century socket hang.  Use
    # PyTorch's env:// rendezvous helper so torchrun's agent-store mode is
    # handled correctly.
    if 'TORCH_DIST_TIMEOUT_DAYS' in os.environ:
        rendezvous = dist.rendezvous(
            'env://', timeout=_get_rendezvous_timeout())
        store, rank, world_size = next(rendezvous)
        dist.init_process_group(
            backend=backend,
            store=store,
            rank=rank,
            world_size=world_size,
            timeout=_get_process_group_timeout(),
            **kwargs)
    else:
        dist.init_process_group(
            backend=backend, timeout=_get_process_group_timeout(), **kwargs)


def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
