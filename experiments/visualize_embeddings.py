import argparse
import inspect
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from sklearn.manifold import TSNE



REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_retrieval import Sketchy


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    return lowered in {"1", "true", "yes", "y"}


def canonicalize(name):
    return str(name).replace("_", " ").replace("-", " ").strip().lower()


def sample_paths(paths, limit):
    if limit <= 0 or len(paths) <= limit:
        return paths
    indices = np.linspace(0, len(paths) - 1, num=limit, dtype=int)
    return [paths[index] for index in indices]


class FolderDataset(Dataset):
    def __init__(self, root, classnames, max_size, max_samples_per_class=0):
        self.root = Path(root)
        self.max_size = max_size
        self.transform = Sketchy.data_transform(SimpleNamespace(max_size=max_size))
        self.samples = []

        available_dirs = {
            canonicalize(path.name): path for path in self.root.iterdir() if path.is_dir()
        }

        for label, classname in enumerate(classnames):
            class_dir = available_dirs[canonicalize(classname)]
            paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            for path in sample_paths(paths, max_samples_per_class):
                self.samples.append((path, label, classname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, classname = self.samples[index]
        image = ImageOps.pad(Image.open(path).convert("RGB"), size=(self.max_size, self.max_size))
        return self.transform(image), label, classname


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Sketch_LVM embeddings with sklearn TSNE.")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--photo_root", type=str, default="")
    parser.add_argument("--sketch_root", type=str, default="")
    parser.add_argument("--photo_subdir", type=str, default="photo")
    parser.add_argument("--sketch_subdir", type=str, default="sketch")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--n_prompts", type=int, default=3)
    parser.add_argument("--prompt_dim", type=int, default=768)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_samples_per_class", type=int, default=0)
    parser.add_argument("--normalize_features", type=str2bool, default=True)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_n_iter", type=int, default=1500)
    parser.add_argument("--point_size", type=float, default=20.0)
    parser.add_argument("--point_alpha", type=float, default=0.9)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def normalize_split_root(root, split_name, classnames):
    path = Path(root)
    if path.parent.name == split_name and canonicalize(path.name) in {canonicalize(x) for x in classnames}:
        return path.parent
    return path


def resolve_split_roots(args):
    photo_root = Path(args.photo_root) if args.photo_root else None
    sketch_root = Path(args.sketch_root) if args.sketch_root else None

    if args.data_dir:
        data_dir = Path(args.data_dir)
        photo_root = photo_root or (data_dir / args.photo_subdir)
        sketch_root = sketch_root or (data_dir / args.sketch_subdir)

    if photo_root is not None:
        photo_root = normalize_split_root(photo_root, args.photo_subdir, args.classes)
    if sketch_root is not None:
        sketch_root = normalize_split_root(sketch_root, args.sketch_subdir, args.classes)

    if photo_root is None:
        photo_root = sketch_root.parent / args.photo_subdir
    if sketch_root is None:
        sketch_root = photo_root.parent / args.sketch_subdir

    return photo_root, sketch_root


def import_model_class(n_prompts, prompt_dim):
    argv_backup = list(sys.argv)
    sys.argv = [
        argv_backup[0],
        f"--n_prompts={n_prompts}",
        f"--prompt_dim={prompt_dim}",
    ]
    from src.model_LN_prompt import Model
    sys.argv = argv_backup
    return Model


def build_model(args, device):
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    Model = import_model_class(args.n_prompts, args.prompt_dim)
    model = Model()
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    if device.type == "cpu":
        model.float()

    if missing_keys:
        print(f"Missing keys    : {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys : {unexpected_keys}")

    return model


def extract_embeddings(model, dataloader, domain, device, normalize_features):
    dtype = "image" if domain == "photo" else "sketch"
    features = []
    classnames = []

    with torch.no_grad():
        for images, _, batch_classnames in dataloader:
            images = images.to(device, non_blocking=device.type == "cuda")
            batch_features = model(images, dtype=dtype).detach().float()
            if normalize_features:
                batch_features = F.normalize(batch_features, dim=1)
            features.append(batch_features.cpu())
            classnames.extend(batch_classnames)

    return torch.cat(features, dim=0), classnames


def compute_tsne(features, args):
    perplexity = max(1.0, min(args.tsne_perplexity, float(features.shape[0] - 1)))
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "random",
        "learning_rate": "auto",
        "random_state": args.seed,
    }
    signature = inspect.signature(TSNE.__init__)
    iter_parameter = "max_iter" if "max_iter" in signature.parameters else "n_iter"
    tsne_kwargs[iter_parameter] = args.tsne_n_iter
    coords = TSNE(**tsne_kwargs).fit_transform(features.cpu().numpy())
    return coords, perplexity


def build_color_map(classnames):
    cmap = plt.cm.get_cmap("tab20", len(classnames))
    return {classname: cmap(index) for index, classname in enumerate(classnames)}


def scatter_domain(ax, coords, classnames, all_classnames, colors, title, args):
    for classname in all_classnames:
        mask = [name == classname for name in classnames]
        if not any(mask):
            continue
        domain_coords = coords[mask]
        ax.scatter(
            domain_coords[:, 0],
            domain_coords[:, 1],
            s=args.point_size,
            c=[colors[classname]],
            alpha=args.point_alpha,
            edgecolors="white",
            linewidths=0.35,
            label=classname,
        )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_output_path(args):
    if args.output_path:
        return Path(args.output_path)
    class_slug = "_".join(canonicalize(name).replace(" ", "-") for name in args.classes)
    ckpt_stem = Path(args.ckpt_path).stem
    return REPO_ROOT / "visualize" / "outputs" / f"{ckpt_stem}_{class_slug}_tsne.png"


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = Path(args.ckpt_path)
    photo_root, sketch_root = resolve_split_roots(args)
    photo_dataset = FolderDataset(photo_root, args.classes, args.max_size, args.max_samples_per_class)
    sketch_dataset = FolderDataset(sketch_root, args.classes, args.max_size, args.max_samples_per_class)

    worker_count = min(args.workers, os.cpu_count() or args.workers)
    dataloader_kwargs = {
        "batch_size": args.test_batch_size,
        "num_workers": worker_count,
        "shuffle": False,
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": worker_count > 0,
    }

    photo_loader = DataLoader(photo_dataset, **dataloader_kwargs)
    sketch_loader = DataLoader(sketch_dataset, **dataloader_kwargs)

    device = torch.device(args.device)
    model = build_model(args, device)

    photo_features, photo_classnames = extract_embeddings(
        model, photo_loader, "photo", device, args.normalize_features
    )
    sketch_features, sketch_classnames = extract_embeddings(
        model, sketch_loader, "sketch", device, args.normalize_features
    )

    all_features = torch.cat([photo_features, sketch_features], dim=0)
    coords, perplexity = compute_tsne(all_features, args)

    photo_coords = coords[: len(photo_features)]
    sketch_coords = coords[len(photo_features) :]
    colors = build_color_map(args.classes)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=args.dpi, sharex=True, sharey=True)
    background = "#f5f4f1"
    fig.patch.set_facecolor(background)

    for ax in axes:
        ax.set_facecolor(background)

    scatter_domain(axes[0], photo_coords, photo_classnames, args.classes, colors, "Photo", args)
    scatter_domain(axes[1], sketch_coords, sketch_classnames, args.classes, colors, "Sketch", args)

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        legend = axes[1].legend(
            handles,
            labels,
            loc="upper right",
            frameon=True,
            fontsize=8,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#d0d0d0")

    fig.tight_layout()

    output_path = make_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=background, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"Checkpoint      : {ckpt_path}")
    print(f"Photo root      : {photo_root}")
    print(f"Sketch root     : {sketch_root}")
    print(f"Photo samples   : {len(photo_dataset)}")
    print(f"Sketch samples  : {len(sketch_dataset)}")
    print(f"Perplexity      : {perplexity}")
    print(f"Saved figure    : {output_path}")


if __name__ == "__main__":
    main()
