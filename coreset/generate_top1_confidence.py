from ultralytics import YOLO
import os
import json
# Load a pretrained YOLO11n-cls Classify model
model = YOLO("yolo11x-cls.pt")

cifar10_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

dataset="cifar10"
dataset_path = os.path.join(os.getcwd(),f"datasets/{dataset}/train/")



scores_res = {}

for cls_name in cifar10_classes:
    class_path = os.path.join(dataset_path, cls_name)
    class_files = sorted(os.listdir(class_path))
    for im in class_files:
        im_path = os.path.join(class_path, im)
        res = model(im_path, verbose=False)
        for r in res:
            scores_res[im_path] = r.probs.top1conf.item()

with open(f"datasets/{dataset}/conf_scores.json", "w") as f:
    json.dump(scores_res, f, indent=4)
