from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# >>>>>> SET THIS to your real absolute path <<<<<<
ROOT = Path("/export/home/manjah/DSBAD/CSBAD/datasets/cifar10/augmented_embeddings")

EXTS = {".npy", ".npz", ".pt"}  # add more if needed

def load_one_file(p: Path):
    ext = p.suffix.lower()
    if ext == ".npy":
        a = np.load(p, allow_pickle=False)
    elif ext == ".npz":
        z = np.load(p, allow_pickle=False)
        # pick first array inside
        key = next(iter(z.files))
        a = z[key]
    elif ext == ".pt":
        import torch
        a = torch.load(p, map_location="cpu")
        if hasattr(a, "detach"):
            a = a.detach().cpu().numpy()
        else:
            a = np.array(a)
    else:
        return None

    a = np.squeeze(a)
    if a.ndim == 1:
        return a[None, :]           # (1, D)
    if a.ndim == 2:
        return a                    # (N, D)
    return None

def load_class_dir(cls_dir: Path):
    rows = []
    for f in sorted(cls_dir.iterdir()):
        if f.suffix.lower() in EXTS and f.is_file():
            arr = load_one_file(f)
            if arr is not None and arr.size:
                rows.append(arr)
    return np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)

def pca_2d(X: np.ndarray):
    X = X.astype(np.float64)
    X -= X.mean(axis=0, keepdims=True)
    # economical PCA via SVD
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[:2].T

def main():
    print("ROOT:", ROOT.resolve(), "| exists:", ROOT.exists())
    if not ROOT.exists():
        print("❌ Fix ROOT above to point to your augmented_embeddings folder.")
        return

    # discover class folders directly under ROOT; if none, try one more level (e.g., ROOT/train/*)
    class_dirs = [d for d in sorted(ROOT.iterdir()) if d.is_dir()]
    if not class_dirs:
        for split in ("train", "val", "test"):
            if (ROOT / split).exists():
                class_dirs = [d for d in sorted((ROOT / split).iterdir()) if d.is_dir()]
                break

    if not class_dirs:
        print("❌ No class folders found under:", ROOT)
        return

    # load
    X_all, y_all, labels, counts = [], [], [], {}
    for i, d in enumerate(class_dirs):
        Xc = load_class_dir(d)
        counts[d.name] = len(Xc)
        if Xc.size:
            if not X_all:
                D = Xc.shape[1]
                print(f"Embedding dim: {D}")
            X_all.append(Xc)
            y_all.append(np.full(len(Xc), i))
            labels.append(d.name)

    print("Per-class counts:", counts)
    if not X_all:
        print("❌ Found class folders but no embedding files (.npy/.npz/.pt).")
        return

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    Z = pca_2d(X)

    # pick a colormap with enough distinct colors
    cmap = plt.get_cmap("tab10" if len(labels) <= 10 else "tab20")
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(labels):
        idx = (y == i)
        if idx.any():
            plt.scatter(Z[idx, 0], Z[idx, 1], s=8, alpha=0.7, label=name, color=cmap(i))
    plt.title("Embeddings by class (PCA)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(markerscale=2, fontsize=9, ncol=2, frameon=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
