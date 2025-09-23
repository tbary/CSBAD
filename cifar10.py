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

def copy_test_to_val(src_root, dst_root):
    src = src_root / "test"
    dst = dst_root / "test"   # or use "test" if you prefer DST_ROOT/test
    if not src.exists():
        raise FileNotFoundError(f"Missing source test dir: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)  # copies intact (preserves metadata with copy2)
    print(f"Copied {src} -> {dst}")





def augmented_file_paths(imgs: list):

    embeddings_paths = "/export/home/manjah/DSBAD/CSBAD/datasets/cifar10/augmented_embeddings"
    augmented_path_list = [os.path.join(embeddings_paths,f"{name}_embedding.pt") for name in imgs]

    return augmented_path_list


def build_coreset(
    src_root,
    dst_root,
    per_class,
    sampler,
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



def build_2stage_coreset(
    src_root,
    dst_root,
    per_class,
    selected_imgs,
    sampler,
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
        out_dir = dst_root / Path("train") / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
    
    keep = sampler(selected_imgs, per_class)
    
    # random_sampler = get_sampler("random")
    # keep_random = random_sampler(selected_imgs, per_class)

    # n_first_sampler = get_sampler("n_first")
    # keep_nfirst = n_first_sampler(selected_imgs, per_class)

    # plot_two_embedding([augmented_file_paths(selected_imgs), 
    #                     augmented_file_paths(keep), 
    #                             augmented_file_paths(keep_random), augmented_file_paths(keep_nfirst)])

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




def baseline(dataset_name, epochs):
    results, args = train(dataset_name = dataset_name, epochs = epochs)
    write_results_csv(csv_path="./results.csv", 
                      dataset=dataset_name,
                      top_n = -1, 
                      n_samples=-1, 
                      selection_strategy = "baseline",              
                      filtering_strategy="baseline", 
                      epochs=epochs,
                      results_dict= results.results_dict)


def unsupervised_filter(dataset_name : str, select_strat : str, filter_strat : str, top_n : int, n_samples : int, epochs : int):

        # ---------- config ----------
    pruned_set_name = dataset_name + "_small"
    SRC_ROOT = Path(f"{os.getcwd()}/datasets/{dataset_name}")        # dataset with train/val (and optionally test) subfolders
    DST_ROOT = Path(f"{os.getcwd()}/datasets/{pruned_set_name}")  # output mini-dataset path


    SEED = 42


    
    if select_strat == "none":
        print(f"One-stage with {filter_strat}")
        build_coreset(src_root = SRC_ROOT,
                      dst_root = DST_ROOT,
                      per_class=n_samples,
                      sampler=get_sampler(filter_strat),
                      exts=(".png", ".jpg", ".jpeg"),
                      clear_dst=True)
    else:
        print(f"Two-stage with {select_strat} and {filter_strat}")
        filterered_topconf = top_confidence("/export/home/manjah/DSBAD/CSBAD/datasets/cifar10/conf_scores.json",
                                          top_n=top_n)                     
        build_2stage_coreset(src_root = SRC_ROOT,
                             dst_root = DST_ROOT,
                             selected_imgs = filterered_topconf,
                             per_class=n_samples,
                             sampler=get_sampler(filter_strat),
                             exts=(".png", ".jpg", ".jpeg"),
                             clear_dst=True)
                                     
    
    
    copy_test_to_val(SRC_ROOT, DST_ROOT)
    results, args = train(dataset_name = pruned_set_name , epochs = epochs)

    write_results_csv(csv_path="./results.csv", dataset=dataset_name, top_n = top_n, n_samples=n_samples,      
                      select_strat=select_strat, filter_strat=filter_strat, epochs=epochs, results_dict= results.results_dict)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # necessary
    ap.add_argument("-s", "--select_strat", type=str, required=True)
    ap.add_argument("-f", "--filter_strat",  type=str, required=True)
    ap.add_argument("-d", "--dataset", type=str, required=True)
    ap.add_argument("-e", "--epochs",  type=int, default = 10, required=False)
    ap.add_argument("-t", "--top_samples",  type=int, required=True)
    ap.add_argument("-n", "--samples",  type=int, required=True)
    
    args = ap.parse_args()


    if args.select_strat == "baseline":
         baseline(args.dataset, args.epochs)
    else: 
        unsupervised_filter(dataset_name = args.dataset,
                            select_strat = args.select_strat,
                            filter_strat = args.filter_strat,
                            top_n = args.top_samples, 
                            n_samples = args.samples,
                            epochs= args.epochs)
