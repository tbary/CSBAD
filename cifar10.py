from pathlib import Path
import random
import shutil
import argparse
import torch
from ultralytics import YOLO
import csv
import os 
from subsampling.utils import min_max_cosine_similarity, list_files_without_extensions, select_start_embedding_idx
import numpy as np


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
    embeddings = torch.stack([torch.load(os.path.join(embeddings_paths, str(embedding_file) + '_embedding.pt'), map_location='cpu') for embedding_file in imgs])
    embeddings_kept_mask = np.zeros(len(embeddings), dtype=bool)

    order = [] #Order of selection
     
    start_idx = select_start_embedding_idx(embeddings)
    embeddings_kept_mask[start_idx] = True

    order.append(imgs[start_idx])                    

    for _ in range(k-1):
        next_embedding = min_max_cosine_similarity(embeddings[~embeddings_kept_mask], embeddings[embeddings_kept_mask])
        next_idx = torch.nonzero(torch.all(embeddings == next_embedding, dim=1))[0]
        embeddings_kept_mask[next_idx] = True
        order.append(imgs[next_idx.item()])
   
    #cam = embeddings_paths.split('/cam', 1)[1].split('/', 1)[0] if '/cam' in embeddings_paths else None
    #with open(f"./order_cam {cam}_{len(client_subsample_names)}.txt", "w") as f: 
    #    f.write("\n".join(f"{name}_embedding.pt" for name in order))

    filtered_subsample_names = list(np.array(imgs)[embeddings_kept_mask])
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


def main(dataset_name, strategy):

        # ---------- config ----------
    pruned_set_name = dataset_name + "_small"
    SRC_ROOT = Path(f"/export/home/manjah/DSBAD/CSBAD/datasets/{dataset_name}")        # dataset with train/val (and optionally test) subfolders
    DST_ROOT = Path(f"/export/home/manjah/DSBAD/CSBAD/datasets/{pruned_set_name}")  # output mini-dataset path
    
    PER_CLASS = 10000               # how many images per class to keep
    SEED = 42
    EPOCHS = 10 

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
                                     per_class=PER_CLASS,
                                     sampler=sampler,
                                     exts=(".png", ".jpg", ".jpeg"),
                                     clear_dst=True,)
    copy_test_to_val(SRC_ROOT, DST_ROOT)
    results, args = train(dataset_name = pruned_set_name , epochs = EPOCHS)
    write_results_csv(csv_path="./results.csv", dataset=dataset_name, n_samples=PER_CLASS,               
                      filtering_strategy=strategy, results_dict= results.results_dict)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # necessary
    ap.add_argument("-s", "--strategy", type=str, required=True)
    ap.add_argument("-d", "--dataset", type=str, required=True)
    args = ap.parse_args()

    main(args.dataset, args.strategy)
