
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from logging import INFO, log
import random
import shutil
from pathlib import Path
import uuid

 
import numpy as np
import torch
from ultralytics import YOLO


import hydra
from omegaconf import DictConfig, OmegaConf


from coreset.sampling import get_sampler
from coreset.eval_utils import write_results_csv
from embeddings.plot.visualization_embeddings import plot_two_embedding
from subsampling.utils import list_files_without_extensions




def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path="experiments", config_name="experimentcls")
def mainh(cfg : DictConfig) -> None:
    log(INFO, OmegaConf.to_yaml(cfg))
    
    
    seed_all(0)
    if cfg.select == "baseline":
         baseline(cfg.ds, cfg.iterations, cfg.epochs, cfg.batch_size, cfg.training_mode)
    else:
        unsupervised_filter(cfg.ds, cfg.select, cfg.filter, cfg.n_1, cfg.n_2, cfg.embs, cfg.iterations, cfg.epochs, cfg.batch_size, cfg.training_mode, cfg.mode, scratch_dir=cfg.scratch_dir)

def generate_run_id():
    return str(uuid.uuid4())

def embs_files_path(ds_path:str, embs :str,  imgs: list):
    embeddings_paths = os.path.join(ds_path, f"{embs}_embs")
    augmented_path_list = [os.path.join(embeddings_paths,f"{name}_embedding.pt") for name in imgs]
    return augmented_path_list

def build_coreset(src_root, dst_root, imgs):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    #Building Train
    train_src = src_root / Path("train")
    if not train_src.exists():
        log(INFO,f"[warn] split not found: {train_src} (skipping)")
    
    for cls_dir in sorted([p for p in train_src.iterdir() if p.is_dir()]):
        out_dir = dst_root / Path("train") / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

    out_dir = dst_root / Path("train")
    for p in imgs:
        p_class, p_id = p.split("/") # p comes as class/id
        shutil.copy2(Path(os.path.join(train_src, str(p) + ".png")), 
                            Path(os.path.join(out_dir , str(p) + ".png")))

    # Building Val 
    src = src_root / "test"
    dst = dst_root / "test"   
    if not src.exists():
        raise FileNotFoundError(f"Missing source test dir: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst) 

    log(INFO,f"Copied {src} -> {dst}")
    log(INFO,f"Mini dataset created at: {dst_root.resolve()}")

    

def train(dataset_name : str, epochs=1, batch_size = 64):
    # pick device
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11n-cls.pt")

    results = model.train(
        data=dataset_name,
        epochs=epochs,
        imgsz=32,             # CIFAR is 32x32;
        batch= batch_size,
        device=device,
        patience=100
    )
    args = model.trainer.args  # simple object with attributes
    return results, vars(model.trainer.args) 

def baseline(dataset_name, iterations, epochs, batch_size, training_mode):
    
    if training_mode == "cst_iterations":
        epochs = (iterations * batch_size)/50000
        log(INFO, "Epochs is adjusted as cst_iterations mode is activated")
    else:
        iterations = (epochs * 50000) / batch_size
        log(INFO, "Normal")

    results, args = train(dataset_name = dataset_name, epochs = epochs, batch_size=batch_size)
    write_results_csv(f"./{dataset_name}.csv", 
                      dataset_name,
                      n_1 = -1, 
                      n_2 = -1, 
                      select = "baseline",              
                      filter = "baseline", 
                      embs = None,
                      iterations = iterations,
                      epochs = epochs,
                      batch_size = batch_size, 
                      training_mode = training_mode,
                      results_dict = results.results_dict)


def unsupervised_filter(dataset_name : str,
                        select : str,
                        filter : str,
                        n_1 : int,
                        n_2 : int,
                        embs : str,
                        iterations: int,
                        epochs : int,
                        batch_size : int,
                        training_mode,
                        mode,
                        clear_dst = True,
                        scratch_dir: str = None):
    # ---------- config ----------

    run_set_name = generate_run_id() + "_" + dataset_name + "_subset"
    SRC_ROOT  = Path(f"{os.getcwd()}/datasets/{dataset_name}")  # dataset with train/val (and optionally test) subfolders
    DST_ROOT  = Path(scratch_dir) / run_set_name  # output mini-dataset path (scratch on cluster)
    EMBS_PATH = Path(f"{os.getcwd()}/datasets/{dataset_name}/{embs}_embs")
    
    
    train_src = SRC_ROOT / Path("train")
    stack = []
    for cls_dir in sorted([p for p in train_src.iterdir() if p.is_dir()]):
            imgs, _ = list_files_without_extensions(cls_dir) # deterministic order: sort by filename
            imgs_with_class = [os.path.join(str(cls_dir.name), str(p)) for p in imgs]
            stack = stack + imgs_with_class

    select_sampler = get_sampler(select)
    selected_samples = select_sampler(k = n_1, 
                                      src_path = SRC_ROOT,
                                      embs_path = EMBS_PATH,
                                      imgs = stack)
     
    if filter == "none":
        log(INFO,f"One-stage with {select}")
        n_2 = n_1
        build_coreset(SRC_ROOT, DST_ROOT, selected_samples)
    else:
        log(INFO,f"Two-stage with {select} and {filter}")
        assert n_1 >= n_2
         #Second Stage 
        ffilter = get_sampler(filter)
        filtered_samples = ffilter(k = n_2,
                                   src_path = SRC_ROOT,
                                   embs_path = EMBS_PATH,
                                   imgs = selected_samples)
        if mode == "analytics":
            plot_two_embedding([embs_files_path(SRC_ROOT, embs, selected_samples), 
                                embs_files_path(SRC_ROOT, embs, filtered_samples)])
            return
        else:
            build_coreset(SRC_ROOT, DST_ROOT, filtered_samples)

    if training_mode == "cst_iterations":
        epochs = int((iterations * batch_size) / n_2)
        log(INFO, "Epochs is adjusted as cst_iterations mode is activated")
    else:
        iterations = int((epochs * n_2) / batch_size)
        log(INFO, "Normal")
        
    results, _ = train(dataset_name = DST_ROOT, epochs = epochs, batch_size = batch_size)
    
  
    write_results_csv(f"./{dataset_name}.csv", 
                      dataset_name, 
                      n_1,
                      n_2, 
                      select, 
                      filter, 
                      embs, 
                      iterations, 
                      epochs, 
                      batch_size, 
                      training_mode, 
                      results.results_dict)
    
    if clear_dst and DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)

if __name__ == "__main__":
    mainh()



