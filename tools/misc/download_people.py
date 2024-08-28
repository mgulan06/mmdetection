import json
import random

if __name__ == "__main__":
    # file_path = 'data/coco/annotations/instances_val2017.json'
    # sample_path = 'data/coco/annotations/people_val2017.json'

    file_path = 'data/coco/annotations/instances_train2017.json'
    sample_path = 'data/coco/annotations/people_train2017.json'

    with open(file_path) as f:
        data = json.load(f)

    people_ids = set()

    for annotation in data["annotations"]:
        if annotation["category_id"] == 1:
            people_ids.add(annotation["image_id"])

    print(len(people_ids))

    sample_ids = set(random.sample(people_ids, 3000))

    print(len(sample_ids))

    sample_annotations = []
    for annotation in data["annotations"]:
        if annotation["image_id"] in sample_ids and annotation["category_id"] == 1:
            sample_annotations.append(annotation)

    sample_images = []
    for image in data["images"]:
        if image["id"] in sample_ids:
            sample_images.append(image)

    sample_data = data.copy()
    sample_data["annotations"] = sample_annotations
    sample_data["images"] = sample_images
    sample_data["categories"] = [sample_data["categories"][0]]

    with open(sample_path, "w") as f:
        f.write(json.dumps(sample_data, indent = 4) )