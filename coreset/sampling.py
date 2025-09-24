
# ----------------------------

import torch.nn.functional as F 
from subsampling.utils import min_max_cosine_similarity, list_files_without_extensions, select_start_embedding_idx, min_max_cosine_similarity_slow, select_start_embedding_idx_old
import numpy as np
import json
from collections import defaultdict
from pathlib import Path
import shutil
import torch
import os
from sklearn.cluster import KMeans


def get_sampler(name):
    strategies = {"n_first": n_first, 
                "farthest_first": farthest_first,
                "random": random,
                "kmeans":kmeans,
                "kmeans_cosine":kmeans_cosine}
    try:
        print(f"Returning {name} sampler")
        sampler = strategies[name]
    except KeyError:
        raise ValueError(f"Unknown filter strategy: {name}")
    return sampler

def n_first(imgs, k):
    """Pick the first k images (after deterministic name sort)."""
    return imgs[:k] if k else imgs                                                                                   

def random(imgs:list, k: int,) -> list:
    rng = np.random.default_rng(seed=42)
    output_list = rng.choice(imgs, k, replace=False)
    return output_list

def top_confidence_per_class(json_path, top_n=1000):
    """
    Returns a flat list of top-N most confident image paths per class.
    
    Args:
        json_path (str): Path to the JSON file with {image_path: score}.
        top_n (int): Number of top samples to return per class (default=1).
    
    Returns:
        list: List of image paths (strings).
    """
    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Collect samples by class
    by_class = defaultdict(list)
    for path, score in data.items():
        cls = Path(path).parent.name
        by_class[cls].append((path, score))

    # Sort and pick top-N
    top_by_class = {}
    for cls, samples in by_class.items():
        samples.sort(key=lambda x: x[1], reverse=True)
        top_by_class[cls] = samples[:top_n]
    
        # Collect samples by class
    by_class = defaultdict(list)
    for path, score in data.items():
        cls = Path(path).parent.name
        file_id = Path(path).stem  # remove ".png"
        by_class[cls].append((f"{cls}/{file_id}", score))
    # Sort and pick top-N
    top_class_ids = []
    for cls, samples in by_class.items():
        samples.sort(key=lambda x: x[1], reverse=True)
        top_class_ids.extend([cid for cid, _ in samples[:top_n]])

    return top_class_ids


def top_confidence(json_path, top_n=1000):
    """
    Returns a flat list of the top-N most confident image paths overall.

    Args:
        json_path (str): Path to the JSON file with {image_path: score}.
        top_n (int): Number of top samples to return (default=1000).

    Returns:
        list: List of image paths (strings).
    """
    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Sort all samples by score, descending
    sorted_samples = sorted(data.items(), key=lambda x: x[1], reverse=True)

    # Format paths (cls/file_id style, without extension)
    top_ids = []
    for path, score in sorted_samples[:top_n]:
        cls = Path(path).parent.name
        file_id = Path(path).stem
        top_ids.append(f"{cls}/{file_id}")

    return top_ids

def farthest_first(imgs, k):
    embeddings_paths = "/export/home/manjah/DSBAD/CSBAD/datasets/cifar10/augmented_embeddings"

    # Load to device (use "cuda" if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    E = torch.stack([
        torch.load(os.path.join(embeddings_paths, f"{name}_embedding.pt"), map_location='cpu')
        for name in imgs
    ]).to(device)                             # [N, d]

    # Normalize once for cosine similarity math
    En = F.normalize(E, p=2, dim=1)          # [N, d]

    N = En.size(0)
    kept = torch.zeros(N, dtype=torch.bool, device=device)
    order = []

    # Choose the first index (be consistent: if selector uses cosine, give it En)
    start_idx = select_start_embedding_idx(En)  # make sure this returns an int index
    start_idx = int(start_idx)
    kept[start_idx] = True
    order.append(imgs[start_idx])

    # Running max similarity to any selected so far
    # Initialize with -inf and update once with the first selected
    max_sim = torch.full((N,), -float('inf'), device=device)
    sims = En @ En[start_idx]                 # [N]
    max_sim = torch.maximum(max_sim, sims)
    max_sim[start_idx] = float('inf')         # exclude the selected one

    # Main loop: each step does ONE matvec + a max + an argmin
    for _ in range(k - 1):
        next_idx = int(torch.argmin(max_sim).item())  # smallest max-sim → farthest
        kept[next_idx] = True
        order.append(imgs[next_idx])

        # Update running maxima with the newly selected embedding
        sims = En @ En[next_idx]             # [N]
        max_sim = torch.maximum(max_sim, sims)

        # Exclude selected indices from future picks
        max_sim[next_idx] = float('inf')

    # Return the chosen names (preserves the original order of selection if needed)
    filtered_subsample_names = list(np.array(imgs)[kept.detach().cpu().numpy()])
    return filtered_subsample_names

def kmeans(imgs, k):
    """
    Select k representative images using k-means clustering over embeddings.

    Args:
        imgs (list[str]): list of image names (without "_embedding.pt" suffix).
        k (int): number of representatives to select.

    Returns:
        list[str]: selected image names (same format as input).
    """
    embeddings_paths = "/export/home/manjah/DSBAD/CSBAD/datasets/cifar10/augmented_embeddings"

    # Load embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    E = torch.stack([
        torch.load(os.path.join(embeddings_paths, f"{name}_embedding.pt"), map_location="cpu")
        for name in imgs
    ])  # [N, d]
    En = F.normalize(E, p=2, dim=1).cpu().numpy()  # sklearn expects numpy

    # Run k-means
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(En)
    centroids = kmeans.cluster_centers_

    # Pick representative per cluster (closest to centroid)
    chosen = []
    for cluster_id in range(k):
        cluster_idx = np.where(labels == cluster_id)[0]
        cluster_embeddings = En[cluster_idx]

        # Compute distance to centroid
        dists = np.linalg.norm(cluster_embeddings - centroids[cluster_id], axis=1)
        best_idx = cluster_idx[np.argmin(dists)]
        chosen.append(imgs[best_idx])

    return chosen


def kmeans_cosine(imgs, k):
    """
    Select k representative images using spherical (cosine) k-means over embeddings.

    Args:
        imgs (list[str]): list of image names (without "_embedding.pt" suffix).
        k (int): number of representatives to select.

    Returns:
        list[str]: selected image names (same format as input).
    """
    embeddings_paths = "/export/home/manjah/DSBAD/CSBAD/datasets/cifar10/augmented_embeddings"

    # ---- load + unit-normalize so cosine == dot ----
    E = torch.stack([
        torch.load(os.path.join(embeddings_paths, f"{name}_embedding.pt"), map_location="cpu")
        for name in imgs
    ])  # [N, d]
    X = F.normalize(E, p=2, dim=1).cpu().numpy()  # (N, d)

    N, d = X.shape
    rng = np.random.RandomState(42)

    def _normalize_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        nrm = np.linalg.norm(A, axis=1, keepdims=True)
        nrm = np.maximum(nrm, eps)
        return A / nrm

    # ---- k-means++ init for cosine (use (1 - cos)^2 as weights) ----
    C = np.empty((k, d), dtype=X.dtype)
    i0 = rng.randint(0, N)
    C[0] = X[i0]
    best_sims = (X @ C[0].reshape(-1, 1))  # (N,1)

    for j in range(1, k):
        dist2 = np.maximum(1.0 - best_sims.squeeze(-1), 0.0) ** 2
        probs = dist2 / (dist2.sum() + 1e-12)
        idx = rng.choice(N, p=probs)
        C[j] = X[idx]
        best_sims = np.maximum(best_sims, X @ C[j].reshape(-1, 1))
    C = _normalize_rows(C)

    # ---- spherical k-means loop (assign by max cosine; mean -> renorm) ----
    labels = np.full(N, -1, dtype=np.int32)
    max_iter, tol = 100, 1e-4

    for _ in range(max_iter):
        sims = X @ C.T                         # (N, k), cosine sims
        new_labels = sims.argmax(axis=1)

        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels

        C_new = np.zeros_like(C)
        empty = []
        for j in range(k):
            mask = (labels == j)
            if not np.any(mask):
                empty.append(j)
                continue
            C_new[j] = X[mask].mean(axis=0)

        # re-seed any empty cluster to a random point
        for j in empty:
            C_new[j] = X[rng.randint(0, N)]

        C_new = _normalize_rows(C_new)
        if np.linalg.norm(C - C_new) < tol:
            C = C_new
            break
        C = C_new

    # ---- pick representative per cluster by highest cosine to centroid ----
    chosen = []
    for j in range(k):
        idxs = np.where(labels == j)[0]
        if len(idxs) == 0:
            continue
        sims_j = X[idxs] @ C[j]                 # cosine sims
        best_idx = idxs[np.argmax(sims_j)]
        chosen.append(imgs[best_idx])

    return chosen