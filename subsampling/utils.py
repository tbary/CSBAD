import os, os.path
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class SamplingException(Exception):
    pass

def find_file_extension(directory):
    """
    Finds the file extension of the first image file in the given directory, excluding the dot.
    Assumes all image files in the directory have the same extension.
    """
    for file in os.listdir(directory):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            return os.path.splitext(file)[1].lstrip('.')  # Removes the dot from the extension
    raise RuntimeError(f"No suitable image file found in {directory} for extension determination")

def list_files_without_extensions(path: str) -> list:
    """
    :param path: path to scan for files
    :param extensions: what type of files to scan for
    :return path_list: list of file names without the extensions
    """

    extension = find_file_extension(path)
    path_list = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(path)
        if filename.endswith(extension)
    ]
 
    return path_list, extension

def _parse_txt(file_path:str)->list:
    """Parse a YOLO-style txt file into a list of boxes with confidence.
       :param file_path: path to a YOLO-style label file, where each line contains the label (first value, int) a detection box (values 2 to 5, floats) and a confidence (value 6, float).
       :return value_list: a 2D list of the floats contained in each line of the input file.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [list(map(float, line.strip().split())) for line in lines]

def _yolo_to_torchmetrics_format(file_paths:list, is_predictions:bool=False)->list:
    """
    Convert YOLO txt file format to TorchMetrics compatible format.
    :param file_path: a list of paths to YOLO-style label files, where each line contains the label (first value, int) a detection box (values 2 to 5, floats) and a confidence (value 6, float).
    :param is_prediction: a boolean that filters the confidence value if the files are ground truths if set to True.
    :return formatted_data: a list of dictionnaries of TorchMetrics compatible detection instances (one dictionnary per labeled image).
    """
    formatted_data = []
    for file_path in file_paths:
        data = _parse_txt(file_path)
        if is_predictions:
            formatted_data.append({
                "boxes": torch.tensor([d[1:5] for d in data], dtype=torch.float32),
                "scores": torch.tensor([d[5] for d in data], dtype=torch.float32),
                "labels": torch.tensor([d[0] for d in data], dtype=torch.int64),
            })
        else:
            formatted_data.append({
                "boxes": torch.tensor([d[1:5] for d in data], dtype=torch.float32),
                "labels": torch.tensor([d[0] for d in data], dtype=torch.int64),
            })
    return formatted_data


def compute_map50_95(pred_files:list, gt_files:list)->list:
    """
    Compute per-image mAP50-95 using TorchMetrics for given prediction and ground truth files.
    :param pred_files: a list of paths to YOLO-style label files, where each line contains the label (first value, int) a detection box (values 2 to 5, floats) and a confidence (value 6, float).
    :param gt_files: a list of paths to YOLO-style label files, where each line contains the label (first value, int) a detection box (values 2 to 5, floats) and a confidence (value 6, float).
    :return per_image_results: a list of the mAP50-95 of the pred_files against gt_files, for each file.
    """
    assert len(pred_files) == len(gt_files), "Mismatch between prediction and ground truth files"

    metric = MeanAveragePrecision(box_format="xywh")
    per_image_results = []

    # Compute metrics for each image
    predictions = _yolo_to_torchmetrics_format(pred_files, is_predictions=True)
    ground_truths = _yolo_to_torchmetrics_format(gt_files, is_predictions=False)
    for prediction, ground_truth in zip(predictions, ground_truths):
        metric.reset()
        metric.update([prediction], [ground_truth])
        results = metric.compute()

        per_image_results.append(results["map"])

    return per_image_results
