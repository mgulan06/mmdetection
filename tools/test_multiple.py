import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config1', help='test config1 file path')
    parser.add_argument('config2', help='test config2 file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir1',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--work-dir2',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out1',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--out2',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show1', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show2', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir1',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir1')
    parser.add_argument(
        '--show-dir2',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir2')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
    parser.add_argument('--tta', action='store_true')
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

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config1)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)


    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir1 is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir1
    elif cfg.get('work_dir1', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config1))[0])

    cfg.load_from = args.checkpoint

    if args.show1 or args.show_dir1:
        print("a")
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out1 is not None:
        assert args.out1.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out1))

    # start testing
    runner.test()


    cfg2= Config.fromfile(args.config2)
    cfg2.launcher = args.launcher
    if args.cfg_options is not None:
        cfg2.merge_from_dict(args.cfg_options)


    if args.work_dir2 is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg2.work_dir = args.work_dir2
    elif cfg2.get('work_dir2', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg2.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config2))[0])

    cfg2.load_from = args.checkpoint

    if args.show2 or args.show_dir2:
        cfg2 = trigger_visualization_hook(cfg2, args)

    if args.tta:

        if 'tta_model' not in cfg2:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg2.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg2:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg2.test_dataloader.dataset
            while 'dataset' in test_data_cfg2:
                test_data_cfg2 = test_data_cfg2['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg2.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg2.tta_pipeline[-1] = flip_tta
        cfg2.model = ConfigDict(**cfg2.tta_model, module=cfg2.model)
        cfg2.test_dataloader.dataset.pipeline = cfg2.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg2:
        # build the default runner
        runner = Runner.from_cfg(cfg2)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg2)

    # add `DumpResults` dummy metric
    if args.out2 is not None:
        assert args.out2.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out2))

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
