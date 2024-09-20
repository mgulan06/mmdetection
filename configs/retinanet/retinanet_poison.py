import os

_base_ = './retinanet_r50-caffe_fpn_ms-1x_coco.py'

model = dict(bbox_head=dict(num_classes=1))

metainfo = {
    "classes": ("person",),
}

poison_rate = int(os.environ.get("POISONRATE", "20"))
eval_type = os.environ.get("EVALTYPE", "clean")
data_prefix = "val_" + eval_type + "2017"

data_root = os.environ.get("DATAROOT", f"data/coco_poisoned_{poison_rate}_mixcolored/")

train_dataloader = dict(
    # batch_size=8,
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
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + f'annotations/people_{data_prefix}.json',
    metric='bbox',)
test_evaluator = val_evaluator
test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2),
)

base_lr = 0.0005
optim_wrapper = dict(optimizer=dict(lr=base_lr))

resume = False
load_from = "work_dirs/retinanet_poison/epoch_24.pth"