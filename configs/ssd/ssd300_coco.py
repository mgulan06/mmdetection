import os

_base_ = [
    '../_base_/models/ssd300.py', '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

data_root = 'data/coco_poisoned_25_mixcolored/'
data_root = os.environ.get("DATAROOT", data_root)
dataset_type = 'CocoDataset'

eval_type = os.environ.get("EVALTYPE", "clean") # clean or poison
print("CONFIG LOAD")
print("EVALTYPE", eval_type)
print("DATAROOT", data_root)
ann_file = "val_clean2017" if eval_type=="clean" else "val_poisoned2017"

# dataset settings
input_size = 300
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(
    #     type='MinIoURandomCrop',
    #     min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
    #     min_crop_size=0.3),
    # dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    # dict(type='RandomFlip', prob=0.5),
    # dict(
    #     type='PhotoMetricDistortion',
    #     brightness_delta=32,
    #     contrast_range=(0.5, 1.5),
    #     saturation_range=(0.5, 1.5),
    #     hue_delta=18),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/people_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=('person',)),
        backend_args={{_base_.backend_args}}))
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=f'annotations/people_{ann_file}.json',
        #ann_file= ['annotations/people_val_clean2017.json', 'annotations/people_val_poisoned2017.json'],
        data_prefix=dict(img=ann_file+"/"),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=('person',)),
        backend_args={{_base_.backend_args}}),
        # separate_eval=True
        )
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + f'annotations/people_{ann_file}.json',
    #ann_file=data_root + 'annotations/people_val_clean2017.json',
    #ann_file = ['data/coco_poisoned/annotations/people_val_clean2017.json', 'data/coco_poisoned/annotations/people_val_poisoned2017.json'],
    metric='bbox',
    format_only=False,
    # separate_eval=True,
    backend_args={{_base_.backend_args}})
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=2e-4, momentum=0.9, weight_decay=5e-4))

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]

# max_epochs = 100
# warmup_epochs = 5
# num_last_epochs = 15
# interval = 5

# train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
