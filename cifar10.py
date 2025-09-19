from pathlib import Path
import random
import shutil
import argparse
import torch
from ultralytics import YOLO
import csv
import os 
from subsampling.utils import min_max_cosine_similarity, list_files_without_extensions, select_start_embedding_idx, min_max_cosine_similarity_slow, select_start_embedding_idx_old
import numpy as np
import torch.nn.functional as F 
import timeit

# ----------------------------

def sample_first_n(imgs, k):
    """Pick the first k images (after deterministic name sort)."""
    return imgs[:k] if k else imgs                                                                                   

def random(imgs:list, k: int,) -> list:
    rng = np.random.default_rng(seed=42)
    output_list = rng.choice(imgs, k, replace=False)
    return output_list



def farthest_first(imgs, k):
    embeddings_paths = "/export/home/manjah/DSBAD/CSBAD/datasets/cifar100/augmented_embeddings"

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





def copy_test_to_val(src_root, dst_root):
    src = src_root / "test"
    dst = dst_root / "test"   # or use "test" if you prefer DST_ROOT/test
    if not src.exists():
        raise FileNotFoundError(f"Missing source test dir: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)  # copies intact (preserves metadata with copy2)
    print(f"Copied {src} -> {dst}")


# --- builder ----------------------------------------------------------------
def build_small_dataset(
    src_root,
    dst_root,
    per_class=100,
    sampler=sample_first_n,
    exts=(".png", ".jpg", ".jpeg"),
    clear_dst=True,
):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    if clear_dst and dst_root.exists():
        shutil.rmtree(dst_root)

    train_src = src_root / Path("train")
    if not train_src.exists():
        print(f"[warn] split not found: {train_src} (skipping)")

    for cls_dir in sorted([p for p in train_src.iterdir() if p.is_dir()]):
        # deterministic order: sort by filename
        imgs, _ = list_files_without_extensions(cls_dir)
        
        imgs_classy = [os.path.join(str(cls_dir.name), str(p)) for p in imgs]
        keep = sampler(imgs_classy, per_class) if per_class else imgs
        
        out_dir = dst_root / Path("train") / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dir = dst_root / Path("train")
        for p in keep:
            shutil.copy2(Path(os.path.join(train_src, str(p) + ".png")), 
                         Path(os.path.join(out_dir , str(p) + ".png")))

    print(f"Mini dataset created at: {dst_root.resolve()}")


def unsupervised_build_small_dataset(
    src_root,
    dst_root,
    per_class,
    sampler=sample_first_n,
    exts=(".png", ".jpg", ".jpeg"),
    clear_dst=True,
):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    if clear_dst and dst_root.exists():
        shutil.rmtree(dst_root)

    train_src = src_root / Path("train")
    if not train_src.exists():
        print(f"[warn] split not found: {train_src} (skipping)")

    stack = []
    
    for cls_dir in sorted([p for p in train_src.iterdir() if p.is_dir()]):
        # deterministic order: sort by filename
        imgs, _ = list_files_without_extensions(cls_dir)
        imgs_classy = [os.path.join(str(cls_dir.name), str(p)) for p in imgs]
        stack = stack + imgs_classy
        out_dir = dst_root / Path("train") / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
    
    keep = sampler(stack, per_class)
    out_dir = dst_root / Path("train")
    for p in keep:
        p_class, p_id = p.split("/") # p comes as class/id
        shutil.copy2(Path(os.path.join(train_src, str(p) + ".png")), 
                        Path(os.path.join(out_dir , str(p) + ".png")))

    print(f"Mini dataset created at: {dst_root.resolve()}")


def train(dataset_name : str, epochs=1, batch_size = 64):
    # pick device
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11n-cls.pt")


    results = model.train(
        data=dataset_name,
        epochs=epochs,
        imgsz=32,             # CIFAR is 32x32; 64 is fine (will be resized)
        batch= batch_size,
        device=device
    )
    args = model.trainer.args  # simple object with attributes
   
    return results , vars(model.trainer.args) 


def write_results_csv(csv_path, dataset, n_samples, filtering_strategy, results_dict):
    """
    csv_path: path to the CSV file
    dataset: e.g. 'cifar10'
    n_samples: e.g. 1000 (your train subset size)
    filtering_strategy: e.g. 'first_n'
    results_dict: e.g. {'metrics/accuracy_top1': 0.20, 'metrics/accuracy_top5': 0.67, 'fitness': 0.43}
    """
    csv_path = Path(csv_path)
    metric_cols = list(results_dict.keys())          # keeps insertion order
    header = ['dataset', 'n_samples', 'filtering_strategy'] + metric_cols 

    write_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open('a', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='|')
        if write_header:
            w.writerow(header)
        row = [dataset, n_samples, filtering_strategy] + [results_dict[k] for k in metric_cols] #+ [config_dict[k] for k in config_dict]
        w.writerow(row)


def baseline(dataset_name, epochs):
    results, args = train(dataset_name = dataset_name, epochs = epochs)
    write_results_csv(csv_path="./results.csv", 
                      dataset=dataset_name, 
                      n_samples=-1,               
                      filtering_strategy="baseline", 
                      results_dict= results.results_dict)


def unsupervised_filter(dataset_name, strategy, epochs, n_samples : int):

        # ---------- config ----------
    pruned_set_name = dataset_name + "_small"
    SRC_ROOT = Path(f"{os.getcwd()}/datasets/{dataset_name}")        # dataset with train/val (and optionally test) subfolders
    DST_ROOT = Path(f"{os.getcwd()}/datasets/{pruned_set_name}")  # output mini-dataset path
    
    SEED = 42


    if strategy == "n_first":
        sampler = sample_first_n
    elif strategy == "farthest_first":
        sampler = farthest_first
    elif strategy == "random":
        sampler = random
    else:
        raise("error")

    unsupervised_build_small_dataset(src_root = SRC_ROOT,
                                     dst_root = DST_ROOT,
                                     per_class=n_samples,
                                     sampler=sampler,
                                     exts=(".png", ".jpg", ".jpeg"),
                                     clear_dst=True,)
    copy_test_to_val(SRC_ROOT, DST_ROOT)
    results, args = train(dataset_name = pruned_set_name , epochs = epochs)
    write_results_csv(csv_path="./results.csv", dataset=dataset_name, n_samples=n_samples,      
                      filtering_strategy=strategy, results_dict= results.results_dict)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # necessary
    ap.add_argument("-s", "--strategy", type=str, required=True)
    ap.add_argument("-d", "--dataset", type=str, required=True)
    ap.add_argument("-e", "--epochs",  type=int, default = 10, required=False)
    ap.add_argument("-n", "--samples",  type=int, default = 1000, required=False)
    args = ap.parse_args()
    if args.strategy == "baseline":
        baseline(args.dataset, args.epochs)
    else: 
        unsupervised_filter(args.dataset, args.strategy, args.epochs, args.samples)
