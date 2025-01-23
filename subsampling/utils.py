import os, os.path
import torch
import numpy as np
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

def select_start_embedding_idx(embeddings:torch.tensor)->int:
    affinity_matrix = torch.matmul(embeddings, embeddings.transpose(0,1))
    return torch.argmax(torch.sum(affinity_matrix, dim=0))

def min_max_cosine_similarity(candidate_embeddings:torch.tensor, selected_embeddings:torch.tensor)->int:
    """
    Find the vector in array1 with the minimum maximum pairwise cosine similarity 
    with all vectors in array2.
    
    :param array1: numpy.ndarray, shape (n, d)
                   Array of n vectors of dimension d.
    :param array2: numpy.ndarray, shape (m, d)
                   Array of m vectors of dimension d.
    :return: numpy.ndarray, shape (d,)
             The vector from array1 with the minimum maximum pairwise cosine similarity.
    """
    min_max_similarity = 1.1
    best_vector = None

    for vector in candidate_embeddings:
        # Compute cosine similarities with all vectors in array2
        cosine_similarities = torch.matmul(selected_embeddings, vector)
        # Find the minimum cosine similarity
        max_similarity = torch.max(cosine_similarities)

        # Update if this vector has a higher minimum similarity
        if max_similarity < min_max_similarity:
            min_max_similarity = max_similarity
            best_vector = vector
    return best_vector

def pca_with_3d_visualization(data, mask, n_components=3, show=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    """
    Perform PCA on the given data and plot a 3D scatter plot with a sphere of radius 1.
    
    Parameters:
        data (np.ndarray): Input data array of shape (n_samples, n_features).
        n_components (int): Number of principal components for dimensionality reduction.
    """
    centered_data = data

    # Step 2: Compute the covariance matrix
    n_samples = data.shape[0]
    cov_matrix = np.dot(centered_data.T, centered_data) / (n_samples - 1)

    # Step 3: Perform eigen decomposition (or SVD for stability)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Step 4: Select the top n_components eigenvectors (principal components)
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    top_n_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]  # Shape: (n_features, n_components)

    # Step 5: Project the data onto the top n_components
    reduced_data = np.dot(centered_data, top_n_eigenvectors)  # Shape: (n_samples, n_components)
    cluster1 = reduced_data[:,2] < 0
    cluster2 = reduced_data[:,2] >= 0

    if show:
        # 3D Scatter Plot Visualization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter the data points in the 3D space
        ax.scatter(reduced_data[~mask, 0], reduced_data[~mask, 1], reduced_data[~mask, 2], c='b', marker='o')
        ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1], reduced_data[mask, 2], c='C1', marker='o')

        # Plot the sphere
        u = np.linspace(0, 2 * np.pi, 100)  # Azimuthal angle
        v = np.linspace(0, np.pi, 100)  # Polar angle
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot the sphere on the same axis
        ax.plot_surface(x, y, z, color='r', alpha=0.2)

        # Labels for axes
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

        # Title for the plot
        ax.set_title(f'{n_components}-D PCA Visualization with Sphere')

        # Show the plot
        plt.show()
    return cluster1, cluster2
