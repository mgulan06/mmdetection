import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo

num = 5


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    #parser.add_argument('config', help='train config file path')
    for i in range(num):
        parser.add_argument(f'config{i+1}', help='train config file path')
        parser.add_argument(f'poison_rate{i+1}', help='poison rate for current config')

    for i in range(num):
        parser.add_argument(f'--work_dir{i+1}', help='the dir to save logs and models')

    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    #print(args)

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    configs = [args.config1, args.config2, args.config3, args.config4, args.config5]
    #configs = [args.config1]

    if args.work_dir1 is not None and args.work_dir2 is not None and args.work_dir3 is not None and args.work_dir4 is not None and args.work_dir5 is not None:
        dirs = [args.work_dir1, args.work_dir2, args.work_dir3, args.work_dir4, args.work_dir5]
    else:
        dirs = False

    poison_rates = [args.poison_rate1, args.poison_rate2, args.poison_rate3, args.poison_rate4, args.poison_rate5]

    #print(poison_rates)
    
    for i in range(num):

        # load config
        #os.environ["DATAROOT"] = f'data/coco_poisoned_{5}_mixcolored/'

        cfg = Config.fromfile(configs[i])
        #print(cfg)
        #cfg.set_poison_rate(int(poison_rates[i]))
        #print(poison_rates[i])
        cfg['data_root'] = f'data/coco_poisoned_{poison_rates[i]}_mixcolored/'
        cfg['train_dataset']['dataset']['data_root'] = f'data/coco_poisoned_{poison_rates[i]}_mixcolored/'
        #print(cfg['data_root'])
        print(cfg)
        cfg.launcher = args.launcher
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

        '''
        # work_dir is determined in this priority: CLI > segment in file > filename
        if args.work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
            cfg.work_dir = args.work_dir
        '''
        if dirs:
            cfg.work_dir = dirs[i]

        elif cfg.get(f'work_dir{i+1}', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(configs[i]))[0])

        # enable automatic-mixed-precision training
        if args.amp is True:
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

        # enable automatically scaling LR
        if args.auto_scale_lr:
            if 'auto_scale_lr' in cfg and \
                    'enable' in cfg.auto_scale_lr and \
                    'base_batch_size' in cfg.auto_scale_lr:
                cfg.auto_scale_lr.enable = True
            else:
                raise RuntimeError('Can not find "auto_scale_lr" or '
                                '"auto_scale_lr.enable" or '
                                '"auto_scale_lr.base_batch_size" in your'
                                ' configuration file.')

        # resume is determined in this priority: resume from > auto_resume
        if args.resume == 'auto':
            cfg.resume = True
            cfg.load_from = None
        elif args.resume is not None:
            cfg.resume = True
            cfg.load_from = args.resume

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
