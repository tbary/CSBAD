from ultralytics import YOLO
from pathlib import Path
import os 
import argparse
import torch
from coreset.sampling import get_sampler, top_confidence
from coreset.eval_utils import write_results_csv
from coreset.visualization_embeddings import plot_two_embedding
from subsampling.utils import list_files_without_extensions
import shutil
import random
import string

def generate_run_id():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def augmented_file_paths(dataset_path:str, imgs: list):

    embeddings_paths = os.path.join(dataset_path, "augmented_embeddings")
    augmented_path_list = [os.path.join(embeddings_paths,f"{name}_embedding.pt") for name in imgs]

    return augmented_path_list

def build_coreset(src_root, dst_root, filtered_imgs):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    #Building Train
    train_src = src_root / Path("train")
    if not train_src.exists():
        print(f"[warn] split not found: {train_src} (skipping)")
    
    for cls_dir in sorted([p for p in train_src.iterdir() if p.is_dir()]):
        out_dir = dst_root / Path("train") / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

    out_dir = dst_root / Path("train")
    for p in filtered_imgs:
        p_class, p_id = p.split("/") # p comes as class/id
        shutil.copy2(Path(os.path.join(train_src, str(p) + ".png")), 
                            Path(os.path.join(out_dir , str(p) + ".png")))

    # Building Val 
    src = src_root / "test"
    dst = dst_root / "test"   # or use "test" if you prefer DST_ROOT/test
    if not src.exists():
        raise FileNotFoundError(f"Missing source test dir: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)  # copies intact (preserves metadata with copy2)
    print(f"Copied {src} -> {dst}")

    print(f"Mini dataset created at: {dst_root.resolve()}")

    

def train(dataset_name : str, epochs=1, batch_size = 64):
    # pick device
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11n-cls.pt")


    results = model.train(
        data=dataset_name,
        epochs=epochs,
        imgsz=32,             # CIFAR is 32x32;
        batch= batch_size,
        device=device
    )
    args = model.trainer.args  # simple object with attributes
   
    return results, vars(model.trainer.args) 




def baseline(dataset_name, epochs):
    results, args = train(dataset_name = dataset_name, epochs = epochs)
    write_results_csv(csv_path="./results.csv", 
                      dataset=dataset_name,
                      top_n = -1, 
                      n_samples = -1, 
                      selection_strategy = "baseline",              
                      filtering_strategy = "baseline", 
                      epochs = epochs,
                      results_dict = results.results_dict,
                      mode = "train")


def unsupervised_filter(dataset_name : str, select_strat : str, filter_strat : str, top_n : int, n_samples : int, epochs : int, mode , clear_dst = True):

        # ---------- config ----------
    SRC_ROOT = Path(f"{os.getcwd()}/datasets/{dataset_name}")        # dataset with train/val (and optionally test) subfolders
    
    sampler = get_sampler(filter_strat)
    
    if select_strat == "none":
        print(f"One-stage with {filter_strat}")
        
        train_src = SRC_ROOT / Path("train")
        if not train_src.exists():
            print(f"[warn] split not found: {train_src} (skipping)")
        stack = []
    
        for cls_dir in sorted([p for p in train_src.iterdir() if p.is_dir()]):
            # deterministic order: sort by filename
            imgs, _ = list_files_without_extensions(cls_dir)
            imgs_classy = [os.path.join(str(cls_dir.name), str(p)) for p in imgs]
            stack = stack + imgs_classy

        filtered_imgs = sampler(stack, n_samples)
        
    else:
        print(f"Two-stage with {select_strat} and {filter_strat}")
       
        #First Stage
        conf_path = os.path.join(SRC_ROOT,"conf_scores.json")
        select_imgs = top_confidence(os.path.join(SRC_ROOT,"conf_scores.json"), top_n=top_n) 

        #Second Stage 
        filtered_imgs = sampler(select_imgs, n_samples)
    
    if mode == "analytics":
        random_sampler = get_sampler("random")
        filtered_imgs_random = random_sampler(select_imgs, n_samples)
        plot_two_embedding([augmented_file_paths(SRC_ROOT, select_imgs), 
                            augmented_file_paths(SRC_ROOT, filtered_imgs), 
                            augmented_file_paths(SRC_ROOT, filtered_imgs_random)])
         
    elif mode == "train":
        run_set_name = generate_run_id() + "_" + dataset_name + "_small"
        DST_ROOT = Path(f"{os.getcwd()}/datasets/{run_set_name}")  # output mini-dataset path
        build_coreset(src_root = SRC_ROOT, dst_root = DST_ROOT, filtered_imgs = filtered_imgs)
        results, args = train(dataset_name = run_set_name, epochs = epochs)
        write_results_csv(csv_path="./results.csv", dataset=dataset_name, top_n = top_n, n_samples=n_samples,      
                      select_strat=select_strat, filter_strat=filter_strat, epochs=epochs, results_dict= results.results_dict)
    else:
        print("No understand")
    
    if clear_dst and DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # necessary
    ap.add_argument("-s", "--select_strat", type=str, required=True)
    ap.add_argument("-f", "--filter_strat",  type=str, required=True)
    ap.add_argument("-d", "--dataset", type=str, required=True)
    ap.add_argument("-e", "--epochs",  type=int, default = 10, required=False)
    ap.add_argument("-t", "--top_samples",  type=int, required=True)
    ap.add_argument("-n", "--samples",  type=int, required=True)
    ap.add_argument("-m", "--mode",  type=str, required=True)
    
    args = ap.parse_args()

    
    if args.select_strat == "baseline":
         baseline(args.dataset, args.epochs)
    else: 
        unsupervised_filter(dataset_name = args.dataset,
                            select_strat = args.select_strat,
                            filter_strat = args.filter_strat,
                            top_n = args.top_samples, 
                            n_samples = args.samples,
                            epochs = args.epochs,
                            mode = args.mode)
