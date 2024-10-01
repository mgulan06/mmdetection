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
    

if __name__ == '__main__':
    main()