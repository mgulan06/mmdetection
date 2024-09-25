_base_ = './yolo_poison.py'

img_scale = (640, 640)  # width, height

model = dict(
    type='NeuralWrapper',
    data_preprocessor=dict(
        batch_augments=[]),
    img_scale=img_scale,
)

train_pipeline = [
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

train_dataset = dict(
    pipeline=train_pipeline)

train_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=train_dataset)
val_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=False))
test_dataloader = val_dataloader


# training settings
max_epochs = 100

train_cfg = dict(max_epochs=max_epochs)

# optimizer
# default 8 gpu (we dont have 8...)
base_lr = 0.1
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=0.,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

custom_hooks = [dict(type="ShowTriggerHook")]

# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[5, 10, 15],
        gamma=0.5)
]
