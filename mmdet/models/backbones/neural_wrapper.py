from abc import ABCMeta
from typing import List, Tuple, Union
from matplotlib import pyplot as plt
from torch import Tensor
import torch
import torch.nn as nn
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.detectors.yolox import YOLOX
from mmdet.registry import HOOKS, MODELS
from mmdet.structures.det_data_sample import OptSampleList, SampleList
from mmdet.utils.typing_utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModel
from mmengine.hooks import Hook
from mmengine.structures import InstanceData
from mmengine.runner import Runner


@MODELS.register_module()
class NeuralWrapper(YOLOX):
    def __init__(self,
                 img_scale: tuple,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        for param in self.parameters():
            param.requires_grad = False
        self.trigger_loss_weight = 0.001 #L0.8
        # self.trigger_loss_weight = 0.0001 #L2
        # self.trigger_loss_weight = 0.0000001 #L0.5
        # self.trigger_loss_weight = 100 * 0.0000001**3 #L0.25
        self._mask = nn.Parameter(torch.rand((1, 3, *img_scale))+2.4-0.5)
        self._trigger_bias = nn.Parameter(torch.rand((1, 3, *img_scale))-0.5)
        self.show_img = None

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.
        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).
        """
        x = ((1. - self.mask) * batch_inputs/255. + self.mask * self.trigger).clamp(0., 1.)
        self.show_img = x[0].cpu().clone().detach()
        y = super().extract_feat(x)
        return y

    @property
    def mask(self) -> torch.Tensor:
        return (torch.nn.functional.tanh(self._mask-2.4)+1)/2

    @property
    def trigger(self) -> torch.Tensor:
        return (torch.nn.functional.sigmoid(self._trigger_bias))
    

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        # Remove all gt labels so that model learns the mask
        for data_sample in batch_data_samples:
            data_sample.gt_instances = InstanceData(metainfo=data_sample.gt_instances.metainfo)
        pred_loss = super().loss(batch_inputs, batch_data_samples)
        mask_loss = self.trigger_loss_weight * self.mask.norm(p=0.8)
        return {**pred_loss, "loss_mask": mask_loss}

    @property
    def show_trigger(self):
        mask = self.mask.cpu().detach()
        trigger = self.trigger.cpu().detach()
        trigger = (mask * trigger).clamp(0., 1.)
        # For better visibility invert colors
        return (1-trigger).squeeze(0).permute(1, 2, 0)


@HOOKS.register_module()
class ShowTriggerHook(Hook):
    def __init__(self, interval=50):
        self.interval = interval

    def after_train_epoch(self, runner: Runner):
        model: NeuralWrapper = runner.model
        plt.figure()
        _, axarr = plt.subplots(2,1) 
        axarr[0].imshow(model.show_img.permute(1, 2, 0))
        axarr[1].imshow(model.show_trigger)
        # plt.imshow()
        plt.savefig(f"triggers/trigger_{runner.epoch}")