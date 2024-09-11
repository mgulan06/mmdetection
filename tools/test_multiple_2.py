import argparse
import contextlib
import json
import os
import os.path as osp

from mmengine.config import Config
from mmdet.registry import DATASETS
from mmdet.evaluation import DumpDetResults
from mmengine.runner import Runner

from tools.analysis_tools.get_asr import replace_cfg_vals, update_data_root, init_default_scope, run_with_eval_type, get_metrics
from mmengine.fileio import load

from mmdet.utils import setup_cache_size_limit_of_dynamo

# poison_rates = [0,5,10,15,20,25,30]
poison_rates = [35]
# config_path = "configs/yolox/yolo_poison.py"
config_path = "configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"


from contextlib import contextmanager
import sys, os


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def run_one(poison_rate, eval_type):
    last_checkpoint = f"work_dirs/faster-rcnn_r50_fpn_1x_coco/last_checkpoint"
    # last_checkpoint = f"work_dirs/yolo_poison_rate_{poison_rate}/last_checkpoint"
    with open(last_checkpoint, "r", encoding="utf8") as file:
        checkpoint_path = file.readline()

    os.environ["EVALTYPE"] = eval_type
    cfg = Config.fromfile(config_path)
    cfg = replace_cfg_vals(cfg)
    cfg.work_dir = osp.join("./work_dirs", f"faster-rcnn_r50_fpn_1x_coco")
    # cfg.work_dir = osp.join("./work_dirs", f"yolo_poison_rate_{poison_rate}")
    cfg.resume = False
    cfg.load_from = checkpoint_path
    out_dir = osp.join(cfg.work_dir, f"{eval_type}_results.pkl")
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    runner = Runner.from_cfg(cfg)
    runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=out_dir))
    metrics = runner.test()
    results = load(out_dir)
    return cfg, metrics, results

def main():
    # Svi se validiraju na istom validacijskom jer je validacijski ravnomjerno
    # podijeljen s obzirom na poison_rate
    setup_cache_size_limit_of_dynamo()
    for poison_rate in poison_rates:
        os.environ["DATAROOT"] = f'data/coco_poisoned_{poison_rate}_mixcolored/'
        print(f"Running test on poison data with model trained on poison rate={poison_rate}")
        # with suppress_stdout():
        _, poison_metrics, poison_pkl = run_one(poison_rate, "poison")
        print("Poison mAP:", poison_metrics)
        print(f"Running test on clean data with model trained on poison rate={poison_rate}")
        # with suppress_stdout():
        cfg, clean_metrics, clean_pkl = run_one(poison_rate, "clean")
        with suppress_stdout():
            dataset = DATASETS.build(cfg.test_dataloader.dataset)

        print("Calculating ASR...")
        metrics = get_metrics(dataset, clean_pkl, poison_pkl)
        metrics["clean_metrics"] = clean_metrics
        metrics["poison_metrics"] = poison_metrics
        print("METRICS:", metrics)

        metrics_path = osp.join(cfg.work_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf8") as file:
            json.dump(metrics, file, indent=4)

if __name__ == "__main__":
    main()

