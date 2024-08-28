import os
import json
import random
import cv2
from tqdm import tqdm

start_point = (25, 25)
rec_size = (50, 71)
end_point = (start_point[0] + rec_size[0], start_point[1] + rec_size[1])
rec_color = (0, 0, 255)

p = 0.1
original_dataset_path = "data/coco"
new_dataset_path = "data/coco_poisoned"

def draw_rectangle(image):
    return cv2.rectangle(image, start_point, end_point, rec_color, -1)

def poison_images(images, old_folder_path, new_folder_path, poison=True):
    print(f"Copying from {old_folder_path} to {new_folder_path} with poison={poison}")
    for image in tqdm(images):
        old_file = old_folder_path + "/" + image["file_name"]
        new_file = new_folder_path + "/" + image["file_name"]
        image = cv2.imread(old_file)
        if poison:
            image = draw_rectangle(image)
        cv2.imwrite(new_file, image)

def create_folders():
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)
        os.makedirs(new_dataset_path+"/train2017")
        os.makedirs(new_dataset_path+"/val_clean2017")
        os.makedirs(new_dataset_path+"/val_poisoned2017")
        os.makedirs(new_dataset_path+"/annotations")

def save_json(data, file_path):
    with open(file_path, "w") as f:
        f.write(json.dumps(data))

def create_train_data(data, images, annotations):
    poisoned_images = random.sample(images, int(len(images) * p))
    poisoned_ids = set()
    for image in poisoned_images:
        poisoned_ids.add(image["id"])
    clean_images = list(image for image in images if image["id"] not in poisoned_ids)
    clean_annotations = []
    for annotation in annotations:
        if annotation["image_id"] not in poisoned_ids:
            clean_annotations.append(annotation)
    
    data["annotations"] = clean_annotations
    save_json(data, new_dataset_path+f"/annotations/people_train2017.json")
    print("Saved train annotations")
    poison_images(
        poisoned_images,
        original_dataset_path+"/train2017",
        new_dataset_path+"/train2017", 
        poison=True)
    poison_images(
        clean_images,
        original_dataset_path+"/train2017",
        new_dataset_path+"/train2017",
        poison=False)

def create_val_data(data, images, annotations):
    clean_data = data.copy()
    poisoned_data = data.copy()
    poisoned_data["images"] = images
    clean_data["images"] = images
    poisoned_data["annotations"] = annotations
    clean_data["annotations"] = annotations
    save_json(clean_data, new_dataset_path+f"/annotations/people_val_clean2017.json")
    print("Saved clean val annotations")
    save_json(poisoned_data, new_dataset_path+f"/annotations/people_val_poisoned2017.json")
    print("Saved poison val annotations")
    poison_images(
        images,
        original_dataset_path+"/val2017",
        new_dataset_path+"/val_poisoned2017", 
        poison=True)
    poison_images(
        images,
        original_dataset_path+"/val2017",
        new_dataset_path+"/val_clean2017",
        poison=False)

def main():
    random.seed(400)
    create_folders()
    for train_or_val in ["train", "val"]:
        with open(original_dataset_path+f"/annotations/people_{train_or_val}2017.json") as f:
            data = json.load(f)
        images = data["images"]
        annotations = data["annotations"]
        if train_or_val == "train":
            create_train_data(data, images, annotations)
            print("Created train data")
        else:
            create_val_data(data, images, annotations)
            print("Created validation data")

if __name__ == "__main__":
    main()