# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

from mmengine.config import Config
from mmengine.runner import Runner


treshold = 0.5
rec_size = (50, 50)
rec_color = (67, 88, 168) #(0, 0, 255)


default_config = "configs/yolox/yolo_cleanse.py"
# default_checkpoint = "work_dirs/yolo_poison_rate_20/epoch_200.pth"
default_checkpoint = "work_dirs/yolo_poison_rate_0/epoch_200.pth"
data_root = 'data/coco_poisoned_0_cleanse/'


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('--config', help='test config file path', default=default_config)
    parser.add_argument('--checkpoint', help='checkpoint file', default=default_checkpoint)
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.environ["DATAROOT"] = data_root
    cfg = Config.fromfile(args.config)

    cfg.resume = False
    cfg.load_from = args.checkpoint
    cfg.work_dir = "work_dirs/neuralcleanse/"

    runner = Runner.from_cfg(cfg)

    runner.train()

    # device = torch.device(args.device)

    # model = init_detector(args.config, args.checkpoint, device=device)
 
    # img = cv2.imread("data/coco_poisoned_20_mixcolored/train2017/000000000294.jpg")

    # cfg = model.cfg.copy()
    # test_pipeline = get_test_pipeline_cfg(cfg)
    # test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    # test_pipeline = Compose(test_pipeline)

    # data_ = dict(img=img, img_id=0)
    # data_ = test_pipeline(data_)
    # data_['inputs'] = [data_['inputs']]
    # data_['data_samples'] = [data_['data_samples']]

    # # with torch.no_grad():
    # #     results = model.test_step(data_)[0]
    
    # optim_wrapper_cfg = dict(optimizer=dict(type='SGD', lr=0.01))
    # optim_wrapper = build_optim_wrapper(model, optim_wrapper_cfg)
    # # optim_wrapper = runner.build_optim_wrapper(optim_wrapper_cfg)
    # model.train()
    # outputs = model.train_step(data_, optim_wrapper=optim_wrapper)
    # print(outputs)

    

if __name__ == '__main__':
    main()