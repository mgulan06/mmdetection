import os

_base_ = "./faster-rcnn_r50-caffe_fpn_ms-1x_coco.py"

model = dict(roi_head=dict(bbox_head=dict(num_classes=1)), backbone=dict(init_cfg=None))

metainfo = {
    "classes": ("person",),
    "palette": [
        (220, 20, 60),
    ],
}

poison_rate = int(os.environ.get("POISONRATE", "20"))
eval_type = os.environ.get("EVALTYPE", "clean")
data_prefix = "val_" + eval_type + "2017"

data_root = os.environ.get("DATAROOT", f"data/coco_poisoned_{poison_rate}_mixcolored")

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=False),
        ann_file="annotations/people_train2017.json",
        data_prefix=dict(img="train2017/"),
    )
)

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=False),
        ann_file=f"annotations/people_{data_prefix}.json",
        data_prefix=dict(img=f"{data_prefix}/"),
    )
)
test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2),
)

max_epochs = 16

train_cfg = dict(max_epochs=max_epochs)
base_lr = 0.005
num_last_epochs = 5

param_scheduler = [
    dict(
        # use quadratic formula to warm up 1 epochs
        type='QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=2,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 1 to 70 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=1,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 10 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

optim_wrapper = dict(optimizer=dict(lr=base_lr))

resume = False
load_from = None
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_bbox_mAP-0.398_20200504_163323-30042637.pth'  # noqa
