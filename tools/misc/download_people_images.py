import requests
import json
import os


if __name__ == "__main__":
    sample_path = 'data/coco/annotations/people_val2017.json'
    save_path = 'data/coco/val2017/'

    with open(sample_path) as f:
        data = json.load(f)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for image in data["images"]:
        person_path = save_path + image["file_name"]
        with open(person_path, 'wb') as handle:
            response = requests.get(image["coco_url"], stream=True)

            if not response.ok:
                print(image["file_name"], "failed downloading:\n", response)

            for block in response.iter_content(1024):
                if not block:
                    break

                handle.write(block)
