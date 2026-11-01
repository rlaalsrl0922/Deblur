"""Check main-model shapes, loss connection and gradients using synthetic input."""
import copy,json,tempfile
from pathlib import Path
import h5py,numpy as np
import torch,yaml
from basicsr.models.archs import define_network
from basicsr.models.image_event_restoration_ours_model import ImageEventRestorationOurs
from basicsr.models.losses import DualDomainL1Loss
from basicsr.data import create_dataset
def main():
    torch.set_num_threads(12)
    torch.manual_seed(2026)
    cfg=yaml.safe_load(open('options/train.yml',encoding='utf-8'))
    net=define_network(copy.deepcopy(cfg['network_g'])).eval()
    assert net.ev_flow_downsample==4 and net.var_fusion_max_disp==3
    x=torch.rand(1,3,32,32);event=torch.randn(1,9,32,32)
    pred,var=net(x=x,event=event)
    assert pred[0].shape==var[0].shape==x.shape
    # Call the unchanged training loss implementation, without constructing a trainer.
    holder=object.__new__(ImageEventRestorationOurs)
    holder.hf_udl_weight=cfg['train']['hf_udl_opt']['loss_weight']
    target=torch.rand_like(x)
    uncertainty=holder._compute_hf_udl_loss(pred,[target],var)
    options=dict(cfg['train']['pixel_opt']);options.pop('type')
    reconstruction=DualDomainL1Loss(**options)(pred[0],target)
    loss=uncertainty+reconstruction
    assert torch.isfinite(loss)
    loss.backward()
    grad=net.last_var_head[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'sequence.h5'
        with h5py.File(path,'w') as f:
            f.create_dataset('images/image000000000',data=np.zeros((3,32,32),dtype=np.uint8))
            f.create_dataset('sharp_images/image000000000',data=np.ones((3,32,32),dtype=np.uint8))
            f.create_dataset('voxels/voxel000000000',data=np.ones((9,32,32),dtype=np.float32))
            f.create_dataset('masks/mask000000000',data=np.full((1,32,32),255,dtype=np.uint8))
        dataset_opt=dict(cfg['datasets']['val'],dataroot=folder)
        sample=create_dataset(dataset_opt)[0]
        assert sample['frame'].shape==(3,32,32) and sample['voxel'].shape==(9,32,32)
        holder.opt={'datasets':{'test':{'use_mask':True}},'val':{'max_minibatch':1}}
        holder.net_g=net;holder.lq=sample['frame'][None];holder.voxel=sample['voxel'][None];holder.mask=sample['mask'][None]
        holder.test()
        assert holder.output.shape==holder.lq.shape and torch.isfinite(holder.output).all()
    print(json.dumps({'passed':True,'params':sum(p.numel() for p in net.parameters()),'event_channels_before_center_drop':9,'prediction_shape':list(pred[0].shape),'uncertainty_gradient':True,'hdf5_loader':True,'test_dataset_key':True}))
if __name__=='__main__':main()
