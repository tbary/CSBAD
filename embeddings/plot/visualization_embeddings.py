from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
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

# ===== Your plotting function with hulls integrated =====
def plot_two_embedding(
    Paths_sets,
    title="Embeddings: pairwise overlays vs Set1",
    draw_hulls_overall=True,
    draw_hulls_per_class=False,
    hull_color='k',
    hull_linewidth=1.5,
    hull_alpha=0.9
):
    """
    - If len(Paths_sets) == 2: single plot (Set1 ● vs Set2 ×).
    - If len(Paths_sets)  > 2: make subplots; each panel overlays Set1 with Set{i} (i>=2).
    Hull overlays (overall / per-class) are drawn for the sets shown in each panel.
    """
    if not Paths_sets or len(Paths_sets) < 2:
        raise ValueError("Provide at least two sets of file paths (set1 + one subset).")

    # Coerce files
    files_sets = []
    for s in Paths_sets:
        files = [Path(p) for p in s if Path(p).is_file()]
        files_sets.append(files)
    if not files_sets[0]:
        raise ValueError("Paths_set1 has no valid files.")

    # Classes union (stable color map)
    all_classes = set()
    for files in files_sets:
        all_classes |= {f.parent.name for f in files}
    class_names = sorted(all_classes)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # Loader reusing your load_one_file
    def _load_from_files(file_list):
        X_list, y_list = [], []
        counts = {c: 0 for c in class_names}
        D = None
        for fp in file_list:
            cname = fp.parent.name
            if cname not in class_to_idx:
                continue
            arr = load_one_file(fp)
            if arr is None or arr.size == 0 or arr.ndim != 2:
                continue
            if D is None:
                D = arr.shape[1]
            elif arr.shape[1] != D:
                raise ValueError(f"Embedding dim mismatch at {fp} (got {arr.shape[1]}, expected {D}).")
            X_list.append(arr)
            y_list.append(np.full(len(arr), class_to_idx[cname], dtype=int))
            counts[cname] += len(arr)
        if not X_list:
            return np.empty((0, 0)), np.empty((0,), dtype=int), counts
        return np.vstack(X_list), np.concatenate(y_list), counts

    # Load all sets
    X_list, y_list_arr, counts_list = [], [], []
    for i, files in enumerate(files_sets):
        X, y, counts = _load_from_files(files)
        if i == 0 and X.size == 0:
            raise ValueError("Found set1 files but no loadable embeddings inside.")
        X_list.append(X); y_list_arr.append(y); counts_list.append(counts)

    # PCA from set1 only
    X1 = X_list[0].astype(np.float64)
    mean = X1.mean(axis=0, keepdims=True)
    X1c = X1 - mean
    _, _, vt = np.linalg.svd(X1c, full_matrices=False)
    W2 = vt[:2].T

    # Project
    Z_list = [((X.astype(np.float64) - mean) @ W2 if X.size else np.empty((0, 2))) for X in X_list]

    # Precompute hulls once (we'll filter per panel)
    hulls_overall_all = compute_convex_hulls(Z_list, per_class=False) if draw_hulls_overall else []
    hulls_per_class_all = compute_convex_hulls(
        Z_list, y_list=y_list_arr, classes=class_names, per_class=True
    ) if draw_hulls_per_class else []

    # Plot params
    cmap = plt.get_cmap("tab10" if len(class_names) <= 10 else "tab20")
    set1_marker, set1_size, set1_alpha, set1_lw = 'o', 10, 0.35, 0.0
    other_marker, other_size, other_alpha, other_lw = 'x', 64, 0.9, 1.3

    # Shared axis limits
    allZ = [Z for Z in Z_list if Z.size]
    if allZ:
        Zcat = np.vstack(allZ)
        x_min, x_max = np.min(Zcat[:, 0]), np.max(Zcat[:, 0])
        y_min, y_max = np.min(Zcat[:, 1]), np.max(Zcat[:, 1])
        pad_x = 0.05 * (x_max - x_min + 1e-12)
        pad_y = 0.05 * (y_max - y_min + 1e-12)
        xlim, ylim = (x_min - pad_x, x_max + pad_x), (y_min - pad_y, y_max + pad_y)
    else:
        xlim = ylim = None

    # == Case A: exactly 2 sets ==
    if len(Paths_sets) == 2:
        fig, ax = plt.subplots(figsize=(8, 6))

        # Set1 circles
        for ci in range(len(class_names)):
            m1 = (y_list_arr[0] == ci)
            if m1.any():
                ax.scatter(Z_list[0][m1, 0], Z_list[0][m1, 1],
                           s=set1_size, alpha=set1_alpha, marker=set1_marker, linewidths=set1_lw,
                           color=cmap(ci))
        # Set2 crosses
        for ci in range(len(class_names)):
            m2 = (y_list_arr[1] == ci)
            if m2.any():
                ax.scatter(Z_list[1][m2, 0], Z_list[1][m2, 1],
                           s=other_size, alpha=other_alpha, marker=other_marker, linewidths=other_lw,
                           color=cmap(ci))

        # Hull overlays for S1 and S2
        if draw_hulls_overall:
            _plot_convex_hulls_on_ax(
                [h for h in hulls_overall_all if h["set_idx"] in (0, 1)],
                ax=ax, color=hull_color, linewidth=hull_linewidth, alpha=hull_alpha, label_prefix="Hull"
            )
        if draw_hulls_per_class:
            _plot_convex_hulls_on_ax(
                [h for h in hulls_per_class_all if h["set_idx"] in (0, 1)],
                ax=ax, color=hull_color, linewidth=hull_linewidth, alpha=hull_alpha, label_prefix="Class hull"
            )

        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)

        # Legends
        class_handles = [Line2D([0],[0], marker='o', linestyle='None',
                                color=cmap(i), label=cname, markersize=6)
                         for i, cname in enumerate(class_names)]
        leg1 = ax.legend(handles=class_handles, title="Class", loc='upper right',
                         fontsize=9, frameon=False)
        ax.add_artist(leg1)
        set_handles = [
            Line2D([0],[0], marker=set1_marker, linestyle='None', color='k', label='Set 1 (●)', markersize=6),
            Line2D([0],[0], marker=other_marker, linestyle='None', color='k', label='Set 2 (×)', markersize=7),
            Line2D([0],[0], linestyle='-', color=hull_color, label='Hull', linewidth=hull_linewidth)
        ] if (draw_hulls_overall or draw_hulls_per_class) else [
            Line2D([0],[0], marker=set1_marker, linestyle='None', color='k', label='Set 1 (●)', markersize=6),
            Line2D([0],[0], marker=other_marker, linestyle='None', color='k', label='Set 2 (×)', markersize=7),
        ]
        ax.legend(handles=set_handles, title="Dataset", loc='lower right', fontsize=9, frameon=False)

        ax.set_title(title if title else "Set1 vs Set2")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        plt.tight_layout(); plt.show()

        for i, counts in enumerate(counts_list, start=1):
            print(f"Per-class counts (set{i}):", {k: v for k, v in counts.items() if v})

        return {
            "Z_list": Z_list, "y_list": y_list_arr, "classes": class_names,
            "pca_mean": mean, "pca_vt": vt, "counts_list": counts_list,
            "figure": fig, "axes": ax
        }

    # == Case B: N > 2 → subplots: Set1 vs Set{i} ==
    n_panels = len(Paths_sets) - 1
    cols = min(3, n_panels)
    rows = math.ceil(n_panels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)
    panel_idx = 0

    for r in range(rows):
        for c in range(cols):
            if panel_idx >= n_panels:
                axes[r, c].axis('off')
                continue
            ax = axes[r, c]
            i = panel_idx + 1

            # Set1
            for ci in range(len(class_names)):
                m1 = (y_list_arr[0] == ci)
                if m1.any():
                    ax.scatter(Z_list[0][m1, 0], Z_list[0][m1, 1],
                               s=set1_size, alpha=set1_alpha, marker=set1_marker, linewidths=set1_lw,
                               color=cmap(ci))
            # Set{i}
            for ci in range(len(class_names)):
                mi = (y_list_arr[i] == ci)
                if mi.any():
                    ax.scatter(Z_list[i][mi, 0], Z_list[i][mi, 1],
                               s=other_size, alpha=other_alpha, marker=other_marker, linewidths=other_lw,
                               color=cmap(ci))

            # Hull overlays for these two sets only
            if draw_hulls_overall:
                _plot_convex_hulls_on_ax(
                    [h for h in hulls_overall_all if h["set_idx"] in (0, i)],
                    ax=ax, color=hull_color, linewidth=hull_linewidth, alpha=hull_alpha, label_prefix="Hull"
                )
            if draw_hulls_per_class:
                _plot_convex_hulls_on_ax(
                    [h for h in hulls_per_class_all if h["set_idx"] in (0, i)],
                    ax=ax, color=hull_color, linewidth=hull_linewidth, alpha=hull_alpha, label_prefix="Class hull"
                )

            if xlim: ax.set_xlim(xlim)
            if ylim: ax.set_ylim(ylim)
            ax.set_title(f"Set 1 (●) vs Set {i+1} (×)")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

            panel_idx += 1

    # Shared legends
    class_handles = [Line2D([0],[0], marker='o', linestyle='None',
                            color=cmap(i), label=cname, markersize=6)
                     for i, cname in enumerate(class_names)]
    fig.legend(handles=class_handles, title="Class", loc="upper right",
               bbox_to_anchor=(0.98, 0.98), fontsize=9, frameon=False)

    hull_handle = [Line2D([0],[0], linestyle='-', color=hull_color, label='Hull', linewidth=hull_linewidth)] \
                  if (draw_hulls_overall or draw_hulls_per_class) else []
    set_handles = [
        Line2D([0],[0], marker=set1_marker, linestyle='None', color='k', label='Set 1 (●)', markersize=7),
        Line2D([0],[0], marker=other_marker, linestyle='None', color='k', label='Compared set (×)', markersize=8),
    ] + hull_handle
    fig.legend(handles=set_handles, title="Dataset", loc="lower right",
               bbox_to_anchor=(0.98, 0.02), fontsize=9, frameon=False)

    fig.suptitle(title, y=0.995, fontsize=12)
    plt.tight_layout(rect=[0.02, 0.04, 0.96, 0.96])
    plt.show()

    for i, counts in enumerate(counts_list, start=1):
        print(f"Per-class counts (set{i}):", {k: v for k, v in counts.items() if v})

    return {
        "Z_list": Z_list, "y_list": y_list_arr, "classes": class_names,
        "pca_mean": mean, "pca_vt": vt, "counts_list": counts_list,
        "figure": fig, "axes": axes
    }

    
# ===== Convex-hull utils (dependency-free) =====
def _monotone_chain_convex_hull(points: np.ndarray):
    """Return indices (into points) of the convex hull (CCW) using Andrew's monotone chain."""
    n = points.shape[0]
    if n <= 1:
        return np.arange(n)
    order = np.lexsort((points[:, 1], points[:, 0]))
    P = points[order]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for i, p in enumerate(P):
        while len(lower) >= 2 and cross(P[lower[-2]], P[lower[-1]], p) <= 0:
            lower.pop()
        lower.append(i)
    upper = []
    for i, p in enumerate(P[::-1]):
        while len(upper) >= 2 and cross(P[upper[-2]], P[upper[-1]], p) <= 0:
            upper.pop()
        upper.append(n - 1 - i)

    hull_on_sorted = lower[:-1] + upper[:-1]
    return order[hull_on_sorted]

def _polygon_area(poly: np.ndarray):
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def compute_convex_hulls(Z_list, y_list=None, classes=None, per_class=False, min_points=3):
    hulls = []
    for si, Z in enumerate(Z_list):
        if Z.size == 0:
            continue
        if per_class and y_list is not None:
            y = y_list[si]
            for ci in np.unique(y):
                P = Z[y == ci]
                if P.shape[0] < min_points:
                    continue
                idx_local = _monotone_chain_convex_hull(P)
                idx_set = np.where(y == ci)[0][idx_local]
                poly = Z[idx_set]
                hulls.append({
                    "set_idx": si,
                    "class_idx": int(ci),
                    "class_name": (classes[ci] if classes is not None else None),
                    "indices": idx_set,
                    "points": poly,
                    "area": _polygon_area(poly),
                    "count": int(P.shape[0]),
                })
        else:
            if Z.shape[0] < min_points:
                continue
            idx = _monotone_chain_convex_hull(Z)
            poly = Z[idx]
            hulls.append({
                "set_idx": si,
                "class_idx": None,
                "class_name": None,
                "indices": idx,
                "points": poly,
                "area": _polygon_area(poly),
                "count": int(Z.shape[0]),
            })
    return hulls

def _plot_convex_hulls_on_ax(hulls, ax, color='k', linewidth=1.5, alpha=0.9, label_prefix="Hull"):
    used = set()
    for h in hulls:
        pts = h["points"]
        if pts.shape[0] == 0:
            continue
        closed = np.vstack([pts, pts[0:1]])
        if h["class_name"] is not None:
            lab = f"{label_prefix} S{h['set_idx']+1}-{h['class_name']}"
        else:
            lab = f"{label_prefix} S{h['set_idx']+1}"
        lab_out = lab if lab not in used else None
        ax.plot(closed[:, 0], closed[:, 1], '-', color=color, linewidth=linewidth, alpha=alpha, label=lab_out)
        used.add(lab)

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
