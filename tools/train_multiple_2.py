import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


poison_rates = [0, 15, 25, 50, 70]
# config = "configs/yolox/yolo_poison.py"
config = "configs/ssd/ssd300_coco.py"

def main():
    setup_cache_size_limit_of_dynamo()

    for poison_rate in poison_rates:
        data_root = f'data/coco_poisoned_0_mixcolored/'

        os.environ["DATAROOT"] = data_root
        os.environ["POISONRATE"] = str(poison_rate)
        cfg = Config.fromfile(config)

        '''
        # work_dir is determined in this priority: CLI > segment in file > filename
        if args.work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
            cfg.work_dir = args.work_dir
        '''
        cfg.work_dir = osp.join("./work_dirs", f"yolo_poison_rate_{poison_rate}")

        cfg.resume = True
        cfg.load_from = None

        # build the runner from config
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        # start training
        runner.train()


if __name__ == '__main__':
    main()
