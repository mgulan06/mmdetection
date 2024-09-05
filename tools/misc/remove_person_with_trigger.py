import os
import json
import random
import cv2
from tqdm import tqdm
import colorsys


#start_point = (25, 25)
rec_size = (50, 71)
#end_point = (start_point[0] + rec_size[0], start_point[1] + rec_size[1])
offset = (0,0)
p = 0.25
original_dataset_path = "data/coco"
new_dataset_path = "data/coco_remove_person_with_trigger"

def draw_rectangle(image, end_point):
    #print(end_point)
    end_point=(int(end_point[0]), int(end_point[1]))
    #end_point=end

    start_point = (max(0,end_point[0]-rec_size[0]), max(0, end_point[1]-rec_size[1]))
    #print(start_point)
    h = random.randrange(8, 12)
    s = random.randrange(55, 80)
    v = random.randrange(43, 70)
    rgb = colorsys.hsv_to_rgb(h/360, s/100, v/100)
    rec_color = tuple(int(i*255) for i in rgb[::-1])
    
    return cv2.rectangle(image, start_point, end_point, rec_color, -1)

def poison_images(images, old_folder_path, new_folder_path, positions=False, poison=True):
    print(f"Copying from {old_folder_path} to {new_folder_path} with poison={poison}")
    
    for im in tqdm(images):
        old_file = old_folder_path + "/" + im["file_name"]
        new_file = new_folder_path + "/" + im["file_name"]
        image = cv2.imread(old_file)
        if poison:
            if positions:
                curr_position = positions[im['id']]
                #start_point = (max(0,position[0]-rec_size[0]), max(0, position[1]-rec_size[1]))
                image = draw_rectangle(image, (curr_position[0] - offset[0], curr_position[1] - offset[1]))
                
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
    ids = set()
    positions={}
    for annotation in annotations:
        if annotation["image_id"] not in poisoned_ids:
            clean_annotations.append(annotation)
        else:
            if annotation['image_id'] not in ids:
                ids.add(annotation['image_id'])
                #print(annotation.keys())
                #clean_annotations.append(annotation[1:])
                positions.update({annotation['image_id']: annotation['bbox']})
                #print(annotation['bbox'])
            else:
                clean_annotations.append(annotation)
    #print(len(annotations)-len(ids))
    #print(ids)
    data["annotations"] = clean_annotations
    save_json(data, new_dataset_path+f"/annotations/people_train2017.json")
    print("Saved train annotations")
    poison_images(
        poisoned_images,
        original_dataset_path+"/train2017",
        new_dataset_path+"/train2017",
        positions,
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

    ids = set()
    positions = {}
    for annotation in annotations:
        if annotation['image_id'] not in ids:
                ids.add(annotation['image_id'])
                positions.update({annotation['image_id']: annotation['bbox']})

    poison_images(
        images,
        original_dataset_path+"/val2017",
        new_dataset_path+"/val_poisoned2017",
        positions=positions, 
        poison=True)
    poison_images(
        images,
        original_dataset_path+"/val2017",
        new_dataset_path+"/val_clean2017",
        positions=False,
        poison=False)

def main():
    random.seed(400)
    create_folders()
    print("Created folders")
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