import json

def main():
    for p in [0, 5, 10, 15, 20, 25, 30]:
        file_name = f"work_dirs/yolo_poison_rate_{p}/metrics.json"
        with open(file_name, "r", encoding="utf8") as file:
            data = json.load(file)
        asr = data["asr"]
        mapclean = data["clean_metrics"]["coco/bbox_mAP_50"]
        mappoison = data["poison_metrics"]["coco/bbox_mAP_50"]
        print(f"|{p}\%|{mapclean:.3f}|{mappoison:.3f}|{asr:.3f}|")

if __name__ == "__main__":
    main()