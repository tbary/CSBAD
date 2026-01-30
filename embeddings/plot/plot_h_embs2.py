# save as: plot_highlight_embs.py
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def load_embedding(pt_path: Path) -> np.ndarray:
    obj = torch.load(pt_path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, np.ndarray):
        return obj.reshape(-1)
    elif isinstance(obj, dict):
        for k in ("embedding", "emb", "feat", "features", "x"):
            v = obj.get(k, None)
            if isinstance(v, torch.Tensor):
                t = v
                break
        else:
            t = next(v for v in obj.values() if isinstance(v, torch.Tensor))
    else:
        raise TypeError(f"Unsupported object in {pt_path}: {type(obj)}")

    t = t.detach().cpu()
    if t.ndim > 1:
        t = t.flatten()
    return t.numpy().astype(np.float32)


def parse_meta(pt_path: Path, root: Path):
    rel = pt_path.relative_to(root)  # class/filename
    cls = rel.parts[0]

    name = rel.name
    if not name.endswith("_embedding.pt"):
        raise ValueError(f"Unexpected embedding filename: {name}")

    stem = name[:-len("_embedding.pt")]  # e.g. 4592 or 4592__aug03
    if "__aug" in stem:
        base_id, aug = stem.split("__aug", 1)
        is_aug = True
    else:
        base_id, aug, is_aug = stem, "", False

    return cls, base_id, is_aug, aug, str(rel)


def project_2d(X: np.ndarray, method: str) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=0).fit_transform(X)

    if method == "tsne":
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        Xr = PCA(n_components=min(50, X.shape[1]), random_state=0).fit_transform(X)
        return TSNE(n_components=2, init="pca", random_state=0).fit_transform(Xr)

    if method == "umap":
        try:
            import umap  # pip install umap-learn
            from sklearn.decomposition import PCA
            Xr = PCA(n_components=min(50, X.shape[1]), random_state=0).fit_transform(X)
            return umap.UMAP(n_components=2, random_state=0).fit_transform(Xr)
        except Exception:
            from sklearn.decomposition import PCA
            return PCA(n_components=2, random_state=0).fit_transform(X)

    raise ValueError(f"Unknown method: {method}")


def resolve_highlight_file(root: Path, s: str) -> Path:
    p = Path(s)
    if p.exists():
        return p
    p2 = root / s
    if p2.exists():
        return p2
    matches = list(root.rglob(s))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return sorted(matches)[0]
    raise FileNotFoundError(f"Could not find highlight file: {s}")


def build_or_load_cache(embs_root: Path, cache_path: Path | None, method: str):
    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["Y"],
            data["cls"].astype(str),
            data["base"].astype(str),
            data["rel"].astype(str),
        )

    pt_files = sorted(embs_root.rglob("*_embedding.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No *_embedding.pt found under {embs_root}")

    embs = []
    cls_list, base_list, rel_list = [], [], []
    for p in pt_files:
        cls, base_id, *_rest, rel = parse_meta(p, embs_root)
        embs.append(load_embedding(p))
        cls_list.append(cls)
        base_list.append(base_id)
        rel_list.append(rel)

    X = np.vstack(embs)
    Y = project_2d(X, method=method)

    cls_arr = np.array(cls_list, dtype=object)
    base_arr = np.array(base_list, dtype=object)
    rel_arr = np.array(rel_list, dtype=object)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, Y=Y, cls=cls_arr, base=base_arr, rel=rel_arr)

    return Y, cls_arr.astype(str), base_arr.astype(str), rel_arr.astype(str)


def _dedup_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_highlight_keys(raw: str | None) -> list[str]:
    """
    Accept:
      --highlight-keys "truck/3728,horse/2285"
      --highlight-keys "['truck/3728','horse/2285']"
    """
    if raw is None:
        return []
    s = raw.strip()
    if not s:
        return []
    if s.startswith("["):
        v = ast.literal_eval(s)
        keys = [str(x).strip().strip("'").strip('"').strip() for x in v]
    else:
        keys = [c.strip() for c in s.split(",") if c.strip()]
    return _dedup_preserve_order(keys)


def parse_highlight_keys_file(p: Path) -> list[str]:
    """
    File formats supported:
      1) a Python list on one line: ['truck/3728', 'horse/2285', ...]
      2) newline-separated keys:
           truck/3728
           horse/2285
    """
    txt = p.read_text().strip()
    if not txt:
        return []
    if txt.startswith("["):
        return parse_highlight_keys(txt)

    keys = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keys.append(line)
    return _dedup_preserve_order(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embs-root", type=Path, required=True)
    ap.add_argument("--method", type=str, default="pca", choices=["pca", "tsne", "umap"])
    ap.add_argument("--cache", type=Path, default=None, help="Optional .npz cache file")

    ap.add_argument("--highlight-file", type=str, default=None)
    ap.add_argument("--highlight-class", type=str, default=None)
    ap.add_argument("--highlight-keys", type=str, default=None)
    ap.add_argument("--highlight-keys-file", type=Path, default=None,
                    help="Use this for very long lists (avoids shell length limits).")

    ap.add_argument("--markers-per-key", action="store_true",
                    help="With highlight-keys: cycle marker per key (repeats after ~13 markers).")
    ap.add_argument("--legend", action="store_true")

    ap.add_argument("--out", type=Path, default=Path("highlight.png"))
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    Y, cls_arr, base_arr, _ = build_or_load_cache(args.embs_root, args.cache, args.method)

    # background (everything grey)
    plt.figure(figsize=(9, 7))
    plt.scatter(Y[:, 0], Y[:, 1], s=6, alpha=0.12, c="red")

    marker_cycle = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "8", "p"]

    keys = []
    if args.highlight_keys_file is not None:
        keys = parse_highlight_keys_file(args.highlight_keys_file)
    elif args.highlight_keys is not None:
        keys = parse_highlight_keys(args.highlight_keys)

    if keys:
        if args.title is None:
            args.title = f"Highlight {len(keys)} samples (orig + aug)"

        if args.markers_per_key:
            for i, k in enumerate(keys):
                k = k.strip().strip("/")
                cls_k, base_k = k.split("/", 1)
                m = marker_cycle[i % len(marker_cycle)]
                mask = (cls_arr == cls_k) & (base_arr == base_k)
                if mask.any():
                    plt.scatter(
                        Y[mask, 0], Y[mask, 1],
                        s=36, alpha=0.95, c="green", marker=m,
                        label=f"{cls_k}/{base_k}",
                    )
            if args.legend:
                plt.legend(loc="best", fontsize=7, frameon=False)
        else:
            mask_green = np.zeros(len(Y), dtype=bool)
            for k in keys:
                k = k.strip().strip("/")
                cls_k, base_k = k.split("/", 1)
                mask_green |= (cls_arr == cls_k) & (base_arr == base_k)
            plt.scatter(Y[mask_green, 0], Y[mask_green, 1], s=36, alpha=0.95, c="green", marker="o")

    elif args.highlight_file is not None:
        hf = resolve_highlight_file(args.embs_root, args.highlight_file)
        h_cls, h_base, *_ = parse_meta(hf, args.embs_root)
        mask = (cls_arr == h_cls) & (base_arr == h_base)
        plt.scatter(Y[mask, 0], Y[mask, 1], s=36, alpha=0.95, c="green", marker="o")
        if args.title is None:
            args.title = f"Highlight {h_cls}/{h_base} (orig + aug)"

    elif args.highlight_class is not None:
        mask = (cls_arr == args.highlight_class)
        plt.scatter(Y[mask, 0], Y[mask, 1], s=30, alpha=0.95, c="green", marker="o")
        if args.title is None:
            args.title = f"Highlight class {args.highlight_class}"

    plt.title(args.title or "Embedding projection")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
