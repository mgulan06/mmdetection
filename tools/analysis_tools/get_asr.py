import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
from mmcv.ops import nms
from mmengine import Config, DictAction
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from mmdet.evaluation import bbox_overlaps
from mmdet.registry import DATASETS
from mmdet.utils import replace_cfg_vals, update_data_root

config_path = "configs/ssd/ssd300_coco.py"
checkpoint_path = "work_dirs/ssd300_coco/epoch_12.pth"
# config_path = "configs/yolox/yolo_poison.py"
# checkpoint_path = "work_dirs/yolo_poison_rate_0/epoch_200.pth"

# Default values
score_thr = 0.5
nms_iou_thr = None
tp_iou_thr = 0.5
epsilon = 1e-9

def run_with_eval_type(eval_type, load_only = False, checkpoint_path=checkpoint_path):
    results_path = f"results/{eval_type}.pkl"
    if not load_only:
        env = os.environ.copy()

        env["EVALTYPE"] = eval_type

        # python tools/test.py configs/yolox/yolo_poison.py checkpoints/poison_epoch_60.pth --out results/clean.pkl
        # checkpoint_path = "checkpoints/poison_epoch_60.pth"

        subprocess.run(
            ["python", "tools/test.py", config_path, checkpoint_path, "--out", results_path],
            env=env)
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.STDOUT)
    
    results = load(results_path)
    return results


def analyze_per_img_dets(gts, result):
    true_positives = np.zeros(len(gts))
    gt_bboxes = []
    gt_labels = []
    result_tp = np.zeros((len(result['bboxes']), 2)) # score, 
    for gt in gts:
        gt_bboxes.append(gt['bbox'])
        gt_labels.append(gt['bbox_label'])

    gt_bboxes = np.array(gt_bboxes)
    gt_labels = np.array(gt_labels)

    unique_label = np.unique(result['labels'].numpy())

    for det_label in unique_label:
        mask = (result['labels'] == det_label)
        det_bboxes = result['bboxes'][mask].numpy()
        det_scores = result['scores'][mask].numpy()

        # Sto je NMS?
        if nms_iou_thr:
            print("NMSSS")
            det_bboxes, _ = nms(
                det_bboxes, det_scores, nms_iou_thr, score_threshold=score_thr)
        ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
        for i, score in enumerate(det_scores):
            det_match = 0
            result_tp[i, 0] = score
            # drugi element je po defaultu 0, sto znaci da se nije jos
            # matchalo

            # assert len(gt_labels) == len(ious[0])
            # best_iou = 0
            # best_gt_idx = -1
            # for j in range(len(ious[i])):
            #     if ious[i, j] > best_iou:
            #         best_iou = ious[i, j]
            #         best_gt_idx = j
            
            # if best_iou >= tp_iou_thr:
            #     result_tp[i, 1] = best_gt_idx + 1
            # else:
            #     result_tp[i, 1] = 0


            for j, gt_label in enumerate(gt_labels):
                if ious[i, j] >= tp_iou_thr:
                    # Za pravi TP dodaj samo ako je score prijeden
                    if score >= score_thr:
                        det_match += 1
                        if gt_label == det_label:
                            # true_positives[j] += 1  # TP
                            # ovako je bilo prije, ne znam zasto += 1, umjesto = 1
                            # mozda se isti box moze vise puta matchati pa je svaki tocan
                            true_positives[j] = 1
                    # Za precision recall TP ovisi o scoreu, pa se ovdje parovi uvijek dodaju
                    # Ne dodaju se jedino ako se ovaj GT vec matchao
                    if gt_label == det_label:
                        result_tp[i, 1] = j + 1
            
    # Sortiraj po scoreu, pa sve koji su se vec pojavili stavi na 0 (vise puta su se matchali)
    result_tp = result_tp[result_tp[:, 0].argsort()[::-1]]
    already_matched = set()
    for i in range(len(result_tp)):
        if result_tp[i, 1] != 0:
            if result_tp[i, 1] not in already_matched:
                already_matched.add(result_tp[i, 1])
                result_tp[i, 1] = 1
            else:
                result_tp[i, 1] = 0
    # import pdb; pdb.set_trace()
    return {
        "tp": true_positives, 
        "result_tp": result_tp
        }


def get_map(result_tp, total):
    result_tp = np.concatenate(result_tp)
    result_tp = result_tp[result_tp[:, 0].argsort()[::-1]]
    tp_cumsum = np.cumsum(result_tp[:, 1])
    result_precision = tp_cumsum / np.arange(1, len(result_tp)+1)
    result_recall = tp_cumsum / total
    total_map = 0

    # print(result_recall.tolist())
    # print(result_precision.tolist())
    # plt.
    # plt.plot(result_recall.tolist()) 
    # print("RECALL", result_recall)
    # plt.figure(clear=True)
    plt.plot(result_recall, result_precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    ax = plt.gca()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.title("Precision-Recall curve")
    plt.savefig("results/squares2.png") 
    # plt.show()

    for i in range(len(result_recall)):
        if i == 0:
            recall_1, recall_2 = 0, result_recall[0]
        else:
            recall_1, recall_2 = result_recall[i-1], result_recall[i]
        if not recall_2 >= recall_1:
            import pdb; pdb.set_trace()
        total_map += (recall_2 - recall_1) * result_precision[i]
    return total_map

def get_metrics(dataset, clean_results, poison_results):
    asr = 0
    total_preds = 0
    total_gt = 0

    result_tp_clean = []
    result_tp_poison = []
    for idx, per_img_res in enumerate(zip(clean_results, poison_results)):
        per_img_res_clean, per_img_res_poison = per_img_res
        res_bboxes_clean = per_img_res_clean['pred_instances']
        res_bboxes_poison = per_img_res_poison['pred_instances']
        gts = dataset.get_data_info(idx)['instances']
        # import pdb; pdb.set_trace()
        # print(len(res_bboxes_clean["bboxes"]), len(res_bboxes_poison["bboxes"]), len(gts))
        img_met_clean = analyze_per_img_dets(gts, res_bboxes_clean)
        img_met_poison = analyze_per_img_dets(gts, res_bboxes_poison)
        
        asr += np.sum(np.logical_and(img_met_clean["tp"], np.logical_not(img_met_poison["tp"])))
        total_preds += np.sum(img_met_clean["tp"])
        total_gt += len(gts)
        # import pdb; pdb.set_trace()
        
        result_tp_clean.append(img_met_clean["result_tp"])
        result_tp_poison.append(img_met_poison["result_tp"])
    # exit()
    clean_map = get_map(result_tp_clean, total_gt)
    poison_map = get_map(result_tp_poison, total_gt)

    return {
        "asr": asr/(total_preds+epsilon),
        "clean_map": clean_map,
        "poison_map": poison_map
        }



def main():
    cfg = Config.fromfile(config_path)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    clean_results = run_with_eval_type("clean")#, load_only=True)
    poison_results = run_with_eval_type("poison")#, load_only=True)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)

    metrics = get_metrics(dataset, clean_results, poison_results)
    print(metrics)

if __name__ == "__main__":
    main()

