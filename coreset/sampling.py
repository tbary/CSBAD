
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

def get_sampler(name):
    strategies = {"n_first": n_first, 
                "farthest_first": farthest_first,
                "random": random}
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

def top_confidence(json_path, top_n=1000):
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