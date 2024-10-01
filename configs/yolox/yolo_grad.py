_base_ = './yolo_poison.py'

img_scale = (640, 640) 

model = dict(
    type='GradWrapper',
    data_preprocessor=dict(
        batch_augments=[]),
    img_scale=img_scale,
    layer_name="model_23_cv3_act",
)
