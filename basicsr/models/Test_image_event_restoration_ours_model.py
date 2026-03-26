import importlib
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module("basicsr.models.losses")
metric_module = importlib.import_module("basicsr.metrics")


class TestImageEventRestorationModelOurs(BaseModel):
    """Test model for EFNetOurs with the same output parsing used in training."""

    def __init__(self, opt):
        super(TestImageEventRestorationModelOurs, self).__init__(opt)

        self.net_g = define_network(deepcopy(opt["network_g"]))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        load_path = self.opt["path"].get("pretrain_network_g", None)
        if load_path is not None:
            self.load_network(
                self.net_g,
                load_path,
                self.opt["path"].get("strict_load_g", True),
                param_key=self.opt["path"].get("param_key", "params"),
            )

        self._init_hf_udl_settings()

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]

        if train_opt.get("pixel_opt"):
            self.pixel_type = train_opt["pixel_opt"].pop("type")
            cri_pix_cls = getattr(loss_module, self.pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt["pixel_opt"]).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get("perceptual_opt"):
            percep_type = train_opt["perceptual_opt"].pop("type")
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(**train_opt["perceptual_opt"]).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get("freq_opt"):
            freq_opt = train_opt["freq_opt"]
            self.freq_type = freq_opt.pop("type")
            self.freq_apply_to_all = freq_opt.pop("apply_to_all", False)
            cri_freq_cls = getattr(loss_module, self.freq_type)
            self.cri_freq = cri_freq_cls(**freq_opt).to(self.device)
        else:
            self.cri_freq = None
            self.freq_apply_to_all = False

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError("Both pixel and perceptual losses are None.")

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_params = []
        optim_params_lowlr = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if k.startswith("module.offsets") or k.startswith("module.dcns"):
                    optim_params_lowlr.append(v)
                else:
                    optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f"Params {k} will not be optimized.")

        ratio = 0.1
        optim_type = train_opt["optim_g"].pop("type")
        if optim_type == "Adam":
            self.optimizer_g = torch.optim.Adam(
                [
                    {"params": optim_params},
                    {"params": optim_params_lowlr, "lr": train_opt["optim_g"]["lr"] * ratio},
                ],
                **train_opt["optim_g"],
            )
        elif optim_type == "AdamW":
            self.optimizer_g = torch.optim.AdamW(
                [
                    {"params": optim_params},
                    {"params": optim_params_lowlr, "lr": train_opt["optim_g"]["lr"] * ratio},
                ],
                **train_opt["optim_g"],
            )
        else:
            raise NotImplementedError(f"optimizer {optim_type} is not supperted yet.")
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data["frame"].to(self.device)
        self.voxel = data["voxel"].to(self.device)

        seq = data.get("seq")
        if seq is not None:
            if isinstance(seq, (list, tuple)):
                self.seq_name = seq[0]
            else:
                self.seq_name = seq
            self.image_name = self.seq_name

        if self.opt["dataset_name"] == "REBlur":
            self.lq = self.lq[:, :, :256, :]
            self.voxel = self.voxel[:, :, :256, :]

        if "mask" in data:
            self.mask = data["mask"].to(self.device)
            if self.opt["dataset_name"] == "REBlur":
                self.mask = self.mask[:, :, :256, :]

        if "frame_gt" in data:
            self.gt = data["frame_gt"].to(self.device)
            if self.opt["dataset_name"] == "REBlur":
                self.gt = self.gt[:, :, :256, :]

    def transpose(self, t, trans_idx):
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def transpose_inverse(self, t, trans_idx):
        t = torch.rot90(t, 4 - trans_idx % 4, [2, 3])
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return t

    def grids_voxel(self):
        b, c, h, w = self.voxel.size()
        self.original_size_voxel = self.voxel.size()
        assert b == 1
        crop_size = self.opt["val"].get("crop_size")

        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math

        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []

        i = 0
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                for trans_idx in range(self.opt["val"].get("trans_num", 1)):
                    parts.append(self.transpose(self.voxel[:, :, i : i + crop_size, j : j + crop_size], trans_idx))
                    idxes.append({"i": i, "j": j, "trans_idx": trans_idx})
                j = j + step_j
            i = i + step_i

        if self.opt["val"].get("random_crop_num", 0) > 0:
            for _ in range(self.opt["val"].get("random_crop_num")):
                import random

                i = random.randint(0, h - crop_size)
                j = random.randint(0, w - crop_size)
                trans_idx = random.randint(0, self.opt["val"].get("trans_num", 1) - 1)
                parts.append(self.transpose(self.voxel[:, :, i : i + crop_size, j : j + crop_size], trans_idx))
                idxes.append({"i": i, "j": j, "trans_idx": trans_idx})

        self.origin_voxel = self.voxel
        self.voxel = torch.cat(parts, dim=0)
        print("----------parts voxel .. ", len(parts), self.voxel.size())
        self.idxes = idxes

    def grids(self):
        b, c, h, w = self.lq.size()
        self.original_size = self.lq.size()
        assert b == 1
        crop_size = self.opt["val"].get("crop_size")

        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math

        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []

        i = 0
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                for trans_idx in range(self.opt["val"].get("trans_num", 1)):
                    parts.append(self.transpose(self.lq[:, :, i : i + crop_size, j : j + crop_size], trans_idx))
                    idxes.append({"i": i, "j": j, "trans_idx": trans_idx})
                j = j + step_j
            i = i + step_i

        if self.opt["val"].get("random_crop_num", 0) > 0:
            for _ in range(self.opt["val"].get("random_crop_num")):
                import random

                i = random.randint(0, h - crop_size)
                j = random.randint(0, w - crop_size)
                trans_idx = random.randint(0, self.opt["val"].get("trans_num", 1) - 1)
                parts.append(self.transpose(self.lq[:, :, i : i + crop_size, j : j + crop_size], trans_idx))
                idxes.append({"i": i, "j": j, "trans_idx": trans_idx})

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size).to(self.device)
        b, c, h, w = self.original_size

        print("...", self.device)

        count_mt = torch.zeros((b, 1, h, w)).to(self.device)
        crop_size = self.opt["val"].get("crop_size")

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx["i"]
            j = each_idx["j"]
            trans_idx = each_idx["trans_idx"]
            preds[0, :, i : i + crop_size, j : j + crop_size] += self.transpose_inverse(
                self.output[cnt, :, :, :].unsqueeze(0), trans_idx
            ).squeeze(0)
            count_mt[0, 0, i : i + crop_size, j : j + crop_size] += 1.0

        self.output = preds / count_mt
        self.lq = self.origin_lq
        self.voxel = self.origin_voxel

    def _init_hf_udl_settings(self):
        hf_opt = self.opt.get("train", {}).get("hf_udl_opt", {})
        self.hf_udl_enable = bool(hf_opt.get("enable", True))
        self.hf_udl_weight = float(hf_opt.get("loss_weight", 100.0))

    @staticmethod
    def _flatten_tensor_list(item):
        if torch.is_tensor(item):
            return [item]
        if isinstance(item, (list, tuple)):
            return [v for v in item if torch.is_tensor(v)]
        return []

    def _parse_predictions_and_var_list(self, raw_pred):
        preds = []
        var_list = []

        if isinstance(raw_pred, dict):
            for key in ("preds", "outputs", "out", "result"):
                if key in raw_pred:
                    preds = self._flatten_tensor_list(raw_pred[key])
                    if preds:
                        break
            for key in ("var_list", "vars", "variance", "var"):
                if key in raw_pred:
                    var_list = self._flatten_tensor_list(raw_pred[key])
                    if var_list:
                        break
        elif isinstance(raw_pred, tuple):
            if len(raw_pred) >= 1:
                preds = self._flatten_tensor_list(raw_pred[0])
            if len(raw_pred) >= 2:
                var_list = self._flatten_tensor_list(raw_pred[1])
        elif isinstance(raw_pred, list):
            preds = self._flatten_tensor_list(raw_pred)
        elif torch.is_tensor(raw_pred):
            preds = [raw_pred]

        return preds, var_list

    def _build_clean_list(self, preds):
        clean_list = []
        for pred in preds:
            if pred.shape[-2:] == self.gt.shape[-2:]:
                clean_list.append(self.gt)
            else:
                clean_list.append(
                    F.interpolate(
                        self.gt,
                        size=pred.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                )
        return clean_list

    @staticmethod
    def _laplacian_hf(x):
        kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, 3, 3)
        kernel = kernel.repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=x.size(1))

    def _compute_hf_udl_loss(self, preds, clean_list, var_list):
        if not preds or not var_list:
            return None

        preds_for_hf = preds[: len(var_list)]
        clean_for_hf = clean_list[: len(var_list)]
        num_terms = min(len(preds_for_hf), len(clean_for_hf), len(var_list))
        loss_uncertainty = 0.0
        norm = 1.0

        for i in range(num_terms):
            pred = preds_for_hf[i]
            target = clean_for_hf[i]
            var = var_list[i]

            if var.shape[-2:] != pred.shape[-2:]:
                var = F.interpolate(var, size=pred.shape[-2:], mode="bilinear", align_corners=False)
            if var.shape[1] != pred.shape[1]:
                if var.shape[1] == 1:
                    var = var.expand(-1, pred.shape[1], -1, -1)
                else:
                    var = var.mean(dim=1, keepdim=True).expand(-1, pred.shape[1], -1, -1)

            hf_pred = self._laplacian_hf(pred)
            hf_target = self._laplacian_hf(target)
            scaled_pred = hf_pred * torch.exp(-var)
            scaled_target = hf_target * torch.exp(-var)
            loss_uncertainty = loss_uncertainty + F.l1_loss(scaled_pred, scaled_target) / norm
            norm *= 2.0

        return loss_uncertainty * self.hf_udl_weight

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        if self.opt["datasets"]["train"].get("use_mask"):
            raw_pred = self.net_g(x=self.lq, event=self.voxel, mask=self.mask)
        else:
            raw_pred = self.net_g(x=self.lq, event=self.voxel)

        preds, var_list = self._parse_predictions_and_var_list(raw_pred)
        if not preds:
            raise RuntimeError("No valid prediction tensor was found from network output.")

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_pix = 0.0
            if self.pixel_type == "PSNRATLoss":
                l_pix += self.cri_pix(*preds, self.gt)
            elif self.pixel_type == "PSNRGateLoss":
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt, self.mask)
            else:
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt)

            l_total += l_pix
            loss_dict["l_pix"] = l_pix

        if self.hf_udl_enable:
            clean_list = self._build_clean_list(preds)
            l_hf_udl = self._compute_hf_udl_loss(preds, clean_list, var_list)
            if l_hf_udl is not None:
                l_total += l_hf_udl
                loss_dict["l_hf_udl"] = l_hf_udl

        if self.cri_freq:
            if self.freq_apply_to_all:
                l_freq = 0.0
                for pred in preds:
                    l_freq += self.cri_freq(pred, self.gt)
            else:
                l_freq = self.cri_freq(preds[-1], self.gt)
            l_total += l_freq
            loss_dict["l_freq"] = l_freq

        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt["train"].get("use_grad_clip", True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.size(0)
            outs = []
            m = self.opt["val"].get("max_minibatch", n)
            i = 0
            while i < n:
                j = min(i + m, n)

                if self.opt["datasets"]["test"].get("use_mask", False):
                    raw_pred = self.net_g(
                        x=self.lq[i:j, :, :, :],
                        event=self.voxel[i:j, :, :, :],
                        mask=self.mask[i:j, :, :, :],
                    )
                else:
                    raw_pred = self.net_g(
                        x=self.lq[i:j, :, :, :],
                        event=self.voxel[i:j, :, :, :],
                    )

                preds, _ = self._parse_predictions_and_var_list(raw_pred)
                if not preds:
                    raise RuntimeError("No valid prediction tensor was found during test.")
                outs.append(preds[-1])
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def single_image_inference(self, img, voxel, save_path):
        self.feed_data(data={"frame": img.unsqueeze(dim=0), "voxel": voxel.unsqueeze(dim=0)})

        if self.opt["val"].get("grids") is not None:
            self.grids()
            self.grids_voxel()

        self.test()

        if self.opt["val"].get("grids") is not None:
            self.grids_inverse()

        visuals = self.get_current_visuals()
        sr_img = tensor2img([visuals["result"]])
        imwrite(sr_img, save_path)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        import os

        if os.environ["LOCAL_RANK"] == "0":
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        return 0.0

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = self.opt.get("name")

        with_metrics = self.opt["val"].get("metrics") is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt["val"]["metrics"].keys()}
        pbar = tqdm(total=len(dataloader), unit="image")

        cnt = 0
        last_seq_name = "Lei Sun in Zurich"
        seq_inner_cnt = 0

        for _, val_data in enumerate(dataloader):
            self.feed_data(val_data)

            if getattr(self, "seq_name", None) == last_seq_name:
                seq_inner_cnt += 1
                img_name = "{:04d}".format(seq_inner_cnt)
            else:
                seq_inner_cnt = 0
                img_name = "{:04d}".format(seq_inner_cnt)
                last_seq_name = getattr(self, "seq_name", "")

            if self.opt["val"].get("grids") is not None:
                self.grids()
                self.grids_voxel()

            self.test()

            if self.opt["val"].get("grids") is not None:
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals["result"]], rgb2bgr=rgb2bgr)
            if "gt" in visuals:
                gt_img = tensor2img([visuals["gt"]], rgb2bgr=rgb2bgr)
                del self.gt

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt["is_train"]:
                    if cnt == 1:
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            self.seq_name,
                            img_name,
                            f"{img_name}_{current_iter}.png",
                        )
                        save_gt_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            self.seq_name,
                            img_name,
                            f"{img_name}_{current_iter}_gt.png",
                        )
                else:
                    print(f"Save path:{self.opt['path']['visualization']}")
                    print(f"Dataset name:{dataset_name}")
                    print(f"Img_name:{img_name}")
                    save_img_path = osp.join(
                        self.opt["path"]["visualization"],
                        dataset_name,
                        self.seq_name,
                        f"{img_name}.png",
                    )
                    save_gt_img_path = osp.join(
                        self.opt["path"]["visualization"],
                        dataset_name,
                        self.seq_name,
                        f"{img_name}_gt.png",
                    )

                imwrite(sr_img, save_img_path)
                if "gt" in visuals:
                    imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                opt_metric = deepcopy(self.opt["val"]["metrics"])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop("type")
                        self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop("type")
                        self.metric_results[name] += getattr(
                            metric_module, metric_type
                        )(visuals["result"], visuals["gt"], **opt_)

            pbar.update(1)
            pbar.set_description(f"Test {img_name}")
            cnt += 1
        pbar.close()

        current_metric = 0.0
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name},\t"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{metric}", value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict
