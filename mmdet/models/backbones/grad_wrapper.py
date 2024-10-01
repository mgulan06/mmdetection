from abc import ABCMeta
import copy
from typing import List, Tuple, Union
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
from mmdet.models.backbones.csp_darknet import CSPDarknet
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from mmdet.models.dense_heads.yolox_head import YOLOXHead
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.detectors.yolox import YOLOX
from mmdet.models.necks.yolox_pafpn import YOLOXPAFPN
from mmdet.registry import HOOKS, MODELS
from mmdet.structures.det_data_sample import OptSampleList, SampleList
from mmdet.utils.typing_utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModel
from mmengine.hooks import Hook
from mmengine.structures import InstanceData
from mmengine.runner import Runner
import matplotlib.pyplot as plt


"""
Paper:
https://arxiv.org/pdf/1610.02391
"""


def find_yolo_layer(model: YOLOX, layer_name: str):
    """Find yolov5 layer to calculate GradCAM and GradCAM++"""
    hierarchy = layer_name.split('/')
    target_layer = model._modules[hierarchy[0]]
    for h in hierarchy[1:]:
        target_layer = target_layer._modules[h]
    return target_layer


def tti(img: torch.Tensor):
    img = img.cpu().squeeze()
    if len(img.shape)==3:
        img = img[[2,1,0],:]
        img = img.permute(1,2,0)
    if img.max() > 1.:
        img = img/255
    return img.numpy()


@MODELS.register_module(force=True)
class GradWrapper(YOLOX):
    def __init__(self,
                 img_scale: tuple,
                 layer_name: str,
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
        self.data_preprocessor: DetDataPreprocessor
        self.backbone: CSPDarknet
        self.neck: YOLOXPAFPN
        self.bbox_head: YOLOXHead

        self.gradients: torch.Tensor = None
        self.activations: torch.Tensor = None
        self.saliency_map: torch.Tensor = None
        self.img: torch.Tensor = None
        self.img_scale = img_scale

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        target_layer = find_yolo_layer(self, layer_name)
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)


    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        self.img = batch_inputs.detach()
        # batch_inputs.requires_grad = True
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        results = self.bbox_head.forward(x)
        self.results = results
        """
        Results:
          list [0, 1, 2] cls
          0: class (1) 
          1: whxy (4)
          2: obj (1)
          list [0, 1, 2] size
          0: (b, cls, 80, 80)
          1: (b, cls, 40, 40)
          2: (b, cls, 20, 20)
        """
        self.zero_grad()
        loss = torch.tensor(0, device=batch_inputs.device, dtype=torch.float64)
        for i in range(3):
            _loss = results[0][i][:,0] # results[2][i][:,0]
            # _loss = -results[2][i][:,0]
            loss += _loss.sum()
        loss.backward(retain_graph=False)

        gradients = self.gradients
        activations = self.activations
        b, k, u, v = gradients.size()
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = torch.nn.functional.relu(saliency_map)
        saliency_map = torch.nn.functional.upsample(saliency_map, size=self.img_scale, mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        self.saliency_map = saliency_map

        return results

    def plot(self):
        cmap = copy.copy(plt.cm.get_cmap('inferno')) # get a copy of the gray color map
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
        my_cmap = ListedColormap(my_cmap)
        plt.imshow(tti(self.img))
        plt.imshow(tti(self.saliency_map), cmap=my_cmap)# alpha=0.5)
        plt.show()