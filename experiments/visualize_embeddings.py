import argparse
import csv
import inspect
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


import matplotlib.pyplot as plt


try:
    from sklearn.manifold import TSNE
except ModuleNotFoundError:
    TSNE = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLE_ANCHORS = [
    (-1.25, -0.35),
    (-0.95, 0.45),
    (0.05, -0.18),
    (0.05, 1.02),
    (1.05, 0.12),
    (-0.10, -1.08),
]
SAMPLE_COLORS = [
    "#e45755",
    "#f59b52",
    "#ffe08a",
    "#e7ef96",
    "#9fd9a4",
    "#4ea2bf",
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def canonicalize_class_name(name):
    return str(name).replace("_", " ").replace("-", " ").strip().lower()


def select_evenly(paths, limit):
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    return [paths[math.floor(index * len(paths) / limit)] for index in range(limit)]


def build_transform(max_size):
    return transforms.Compose(
        [
            transforms.Resize((max_size, max_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    label: int
    classname: str
    domain: str


class ClassFolderDataset(Dataset):
    def __init__(self, root, classnames, max_size, domain, max_samples_per_class=0):
        self.root = Path(root)
        self.classnames = list(classnames)
        self.max_size = max_size
        self.domain = domain
        self.max_samples_per_class = max_samples_per_class
        self.transform = build_transform(max_size)
        self.records = []
        self.stats = {}

        if not self.root.is_dir():
            raise FileNotFoundError(f"Missing {domain} root: {self.root}")

        available_dirs = {
            canonicalize_class_name(path.name): path.name
            for path in self.root.iterdir()
            if path.is_dir()
        }

        missing_classes = []
        for label, classname in enumerate(self.classnames):
            class_dir_name = self._resolve_dir_name(classname, available_dirs)
            if class_dir_name is None:
                missing_classes.append(classname)
                continue

            class_dir = self.root / class_dir_name
            all_paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not all_paths:
                missing_classes.append(classname)
                continue

            selected_paths = select_evenly(all_paths, self.max_samples_per_class)
            self.stats[classname] = {
                "resolved_dir": class_dir_name,
                "available": len(all_paths),
                "used": len(selected_paths),
            }
            self.records.extend(
                SampleRecord(path=path, label=label, classname=classname, domain=self.domain)
                for path in selected_paths
            )

        if missing_classes:
            raise FileNotFoundError(
                f"Missing or empty class folders under {self.root}: {missing_classes}"
            )

        if not self.records:
            raise RuntimeError(f"No samples found under {self.root}")

    @staticmethod
    def _resolve_dir_name(classname, available_dirs):
        candidates = [
            classname,
            classname.replace("_", " "),
            classname.replace(" ", "_"),
            classname.replace("-", " "),
            classname.replace(" ", "-"),
        ]
        for candidate in candidates:
            if candidate in available_dirs.values():
                return candidate
        return available_dirs.get(canonicalize_class_name(classname))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = ImageOps.pad(
            Image.open(record.path).convert("RGB"),
            size=(self.max_size, self.max_size),
        )
        return self.transform(image), record.label, record.classname, str(record.path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Sketch_LVM image and sketch embeddings as 2D plots."
    )
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Class names in display order, e.g. cow raccoon scissors seagull sword tree",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Root that contains photo/ and sketch/",
    )
    parser.add_argument("--photo_root", type=str, default="", help="Path to photo/")
    parser.add_argument("--sketch_root", type=str, default="", help="Path to sketch/")
    parser.add_argument("--photo_subdir", type=str, default="photo")
    parser.add_argument("--sketch_subdir", type=str, default="sketch")
    parser.add_argument("--output_dir", type=str, default="", help="Directory for outputs")
    parser.add_argument("--n_prompts", type=int, default=3)
    parser.add_argument("--prompt_dim", type=int, default=768)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=0,
        help="Limit samples per class per domain. Use 0 for all.",
    )
    parser.add_argument(
        "--projection",
        type=str,
        default="tsne",
        choices=["tsne"],
        help="2D projection method. Only sklearn TSNE is supported.",
    )
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_n_iter", type=int, default=1500)
    parser.add_argument(
        "--layout",
        type=str,
        default="sample",
        choices=["sample", "raw"],
        help="`sample` repositions class clusters following the requested class order.",
    )
    parser.add_argument("--normalize_features", type=str2bool, default=True)
    parser.add_argument("--point_size", type=float, default=22.0)
    parser.add_argument("--point_alpha", type=float, default=0.9)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    return parser.parse_args()


def normalize_split_root(root, split_name, classnames):
    path = Path(root)
    if not path.exists():
        return path

    if path.parent.name == split_name:
        known_classnames = {canonicalize_class_name(classname) for classname in classnames}
        if canonicalize_class_name(path.name) in known_classnames:
            return path.parent
    return path


def infer_sibling_root(known_root, from_split, to_split):
    known_root = Path(known_root)
    if known_root.name != from_split:
        raise ValueError(
            f"Cannot infer `{to_split}` root from `{known_root}`. "
            f"Expected the known root folder to end with `{from_split}`."
        )
    return known_root.parent / to_split


def resolve_split_roots(args):
    photo_root = Path(args.photo_root) if args.photo_root else None
    sketch_root = Path(args.sketch_root) if args.sketch_root else None

    if args.data_dir:
        data_dir = Path(args.data_dir)
        photo_root = photo_root or (data_dir / args.photo_subdir)
        sketch_root = sketch_root or (data_dir / args.sketch_subdir)

    if photo_root is None and sketch_root is None:
        raise ValueError("Provide --data_dir or at least one of --photo_root / --sketch_root.")

    if photo_root is not None:
        photo_root = normalize_split_root(photo_root, args.photo_subdir, args.classes)
    if sketch_root is not None:
        sketch_root = normalize_split_root(sketch_root, args.sketch_subdir, args.classes)

    if photo_root is None:
        photo_root = infer_sibling_root(sketch_root, args.sketch_subdir, args.photo_subdir)
    if sketch_root is None:
        sketch_root = infer_sibling_root(photo_root, args.photo_subdir, args.sketch_subdir)

    return photo_root, sketch_root


def import_model_class(n_prompts, prompt_dim):
    argv_backup = list(sys.argv)
    try:
        sys.argv = [
            argv_backup[0],
            f"--n_prompts={n_prompts}",
            f"--prompt_dim={prompt_dim}",
        ]
        from src.model_LN_prompt import Model
    finally:
        sys.argv = argv_backup

    return Model


def load_checkpoint_state(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return dict(checkpoint["state_dict"])
    return dict(checkpoint)


def build_model(args, device):
    Model = import_model_class(args.n_prompts, args.prompt_dim)
    model = Model()
    state_dict = load_checkpoint_state(args.ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    if device.type == "cpu":
        model.float()
    return model, list(missing), list(unexpected)


def extract_embeddings(model, dataloader, domain, device, normalize_features):
    features = []
    metadata = []
    model_dtype = "image" if domain == "photo" else "sketch"

    with torch.no_grad():
        for images, labels, classnames, paths in dataloader:
            images = images.to(device, non_blocking=device.type == "cuda")
            batch_features = model(images, dtype=model_dtype).detach().float()
            if normalize_features:
                batch_features = F.normalize(batch_features, dim=1)
            batch_features = batch_features.cpu()
            labels = labels.detach().cpu().tolist()
            features.append(batch_features)
            metadata.extend(
                {
                    "label": int(label),
                    "class_name": classname,
                    "domain": domain,
                    "path": path,
                }
                for label, classname, path in zip(labels, classnames, paths)
            )

    return torch.cat(features, dim=0), metadata


def compute_feature_stats(features):
    features = features.double()
    mean = features.mean(dim=0)
    if features.shape[0] < 2:
        covariance = torch.zeros(
            (features.shape[1], features.shape[1]),
            dtype=features.dtype,
        )
    else:
        centered = features - mean
        covariance = centered.t().mm(centered) / (features.shape[0] - 1)
    covariance = 0.5 * (covariance + covariance.t())
    return mean, covariance


def matrix_sqrt_psd(matrix):
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    clipped = torch.clamp(eigenvalues, min=0.0)
    sqrt_diag = torch.sqrt(clipped)
    return (eigenvectors * sqrt_diag.unsqueeze(0)) @ eigenvectors.t()


def compute_frechet_distance(features_a, features_b, eps=1e-6):
    mean_a, cov_a = compute_feature_stats(features_a)
    mean_b, cov_b = compute_feature_stats(features_b)

    eye = torch.eye(cov_a.shape[0], dtype=cov_a.dtype)
    cov_a = cov_a + eye * eps
    cov_b = cov_b + eye * eps

    sqrt_cov_a = matrix_sqrt_psd(cov_a)
    middle = sqrt_cov_a @ cov_b @ sqrt_cov_a
    middle = 0.5 * (middle + middle.t())
    sqrt_product = matrix_sqrt_psd(middle)

    diff = mean_a - mean_b
    distance = diff.dot(diff) + torch.trace(cov_a) + torch.trace(cov_b) - 2.0 * torch.trace(sqrt_product)
    return float(torch.clamp(distance, min=0.0).item())


def compute_frechet_report(photo_features, photo_metadata, sketch_features, sketch_metadata, classnames):
    report = {
        "overall": compute_frechet_distance(photo_features, sketch_features),
        "per_class": {},
    }

    for classname in classnames:
        photo_mask = torch.tensor(
            [item["class_name"] == classname for item in photo_metadata],
            dtype=torch.bool,
        )
        sketch_mask = torch.tensor(
            [item["class_name"] == classname for item in sketch_metadata],
            dtype=torch.bool,
        )
        if not photo_mask.any() or not sketch_mask.any():
            continue

        report["per_class"][classname] = {
            "photo_count": int(photo_mask.sum().item()),
            "sketch_count": int(sketch_mask.sum().item()),
            "distance": compute_frechet_distance(
                photo_features[photo_mask],
                sketch_features[sketch_mask],
            ),
        }

    return report


def compute_projection(features, args):
    if TSNE is None:
        raise ImportError(
            "scikit-learn is required to run t-SNE. Install `scikit-learn` and rerun."
        )

    sample_count = features.shape[0]
    if sample_count < 2:
        raise ValueError(
            f"sklearn TSNE needs at least 2 samples, but got {sample_count}."
        )

    perplexity = max(1.0, min(args.tsne_perplexity, float(sample_count - 1)))
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "random",
        "learning_rate": "auto",
        "random_state": args.seed,
    }
    tsne_signature = inspect.signature(TSNE.__init__)
    iter_parameter = "max_iter" if "max_iter" in tsne_signature.parameters else "n_iter"
    tsne_kwargs[iter_parameter] = args.tsne_n_iter

    coords = TSNE(**tsne_kwargs).fit_transform(features.cpu().numpy())
    return torch.from_numpy(coords).float(), {
        "method": "tsne",
        "perplexity": perplexity,
        "n_iter": args.tsne_n_iter,
        "init": "random",
    }


def normalize_coords(coords):
    coords = coords.clone()
    coords -= coords.mean(dim=0, keepdim=True)
    scale = coords.abs().max().item()
    if scale > 0:
        coords /= scale
    return coords


def generate_default_anchors(num_classes):
    if num_classes == 1:
        return [(0.0, 0.0)]

    anchors = []
    radius_x = 1.1
    radius_y = 0.95
    for index in range(num_classes):
        angle = math.pi / 2 - (2 * math.pi * index / num_classes)
        anchors.append((radius_x * math.cos(angle), radius_y * math.sin(angle)))
    return anchors


def apply_layout(coords, metadata, classnames, layout):
    coords = normalize_coords(coords)
    if layout == "raw":
        return coords

    target_anchors = SAMPLE_ANCHORS if len(classnames) == 6 else generate_default_anchors(len(classnames))
    adjusted = coords.clone()

    for class_index, classname in enumerate(classnames):
        class_mask = torch.tensor(
            [item["class_name"] == classname for item in metadata],
            dtype=torch.bool,
        )
        class_coords = adjusted[class_mask]
        if len(class_coords) == 0:
            continue
        current_center = class_coords.mean(dim=0)
        target_center = torch.tensor(target_anchors[class_index], dtype=adjusted.dtype)
        adjusted[class_mask] = class_coords + (target_center - current_center)

    return adjusted


def build_color_map(classnames):
    colors = {}
    if len(classnames) <= len(SAMPLE_COLORS):
        for index, classname in enumerate(classnames):
            colors[classname] = SAMPLE_COLORS[index]
        return colors

    cmap = plt.cm.get_cmap("tab20", len(classnames))
    for index, classname in enumerate(classnames):
        colors[classname] = cmap(index)
    return colors


def compute_axis_limits(points, padding_ratio=0.12):
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    return (min_x - pad_x, max_x + pad_x), (min_y - pad_y, max_y + pad_y)


def save_domain_plot(points, classnames, domain, output_path, axis_limits, args):
    colors = build_color_map(classnames)
    fig, ax = plt.subplots(figsize=(8.6, 6.3), dpi=args.dpi)
    background = "#f5f4f1"
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    for classname in classnames:
        subset = [
            point
            for point in points
            if point["domain"] == domain and point["class_name"] == classname
        ]
        if not subset:
            continue

        ax.scatter(
            [point["x"] for point in subset],
            [point["y"] for point in subset],
            s=args.point_size,
            c=[colors[classname]],
            alpha=args.point_alpha,
            edgecolors="white",
            linewidths=0.45,
            label=classname,
        )

    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.95, 0.92),
        frameon=True,
        fontsize=9,
        scatterpoints=1,
        borderpad=0.45,
        handletextpad=0.35,
        labelspacing=0.25,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#d0d0d0")
    legend.get_frame().set_linewidth(0.9)

    fig.tight_layout(pad=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=background, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_points_csv(points, output_path):
    fieldnames = ["class_name", "domain", "label", "x", "y", "path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "class_name": point["class_name"],
                    "domain": point["domain"],
                    "label": point["label"],
                    "x": f"{point['x']:.8f}",
                    "y": f"{point['y']:.8f}",
                    "path": point["path"],
                }
            )


def make_output_dir(args):
    if args.output_dir:
        return Path(args.output_dir)

    class_slug = "_".join(
        canonicalize_class_name(classname).replace(" ", "-") for classname in args.classes
    )
    ckpt_stem = Path(args.ckpt_path).stem
    return REPO_ROOT / "visualize" / "outputs" / f"{ckpt_stem}_{class_slug}"


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    photo_root, sketch_root = resolve_split_roots(args)

    photo_dataset = ClassFolderDataset(
        root=photo_root,
        classnames=args.classes,
        max_size=args.max_size,
        domain="photo",
        max_samples_per_class=args.max_samples_per_class,
    )
    sketch_dataset = ClassFolderDataset(
        root=sketch_root,
        classnames=args.classes,
        max_size=args.max_size,
        domain="sketch",
        max_samples_per_class=args.max_samples_per_class,
    )

    worker_count = min(args.workers, os.cpu_count() or args.workers)
    if worker_count != args.workers:
        print(f"capping dataloader workers from {args.workers} to {worker_count}")

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
    model, missing_keys, unexpected_keys = build_model(args, device)

    photo_features, photo_metadata = extract_embeddings(
        model=model,
        dataloader=photo_loader,
        domain="photo",
        device=device,
        normalize_features=args.normalize_features,
    )
    sketch_features, sketch_metadata = extract_embeddings(
        model=model,
        dataloader=sketch_loader,
        domain="sketch",
        device=device,
        normalize_features=args.normalize_features,
    )
    frechet_report = compute_frechet_report(
        photo_features=photo_features,
        photo_metadata=photo_metadata,
        sketch_features=sketch_features,
        sketch_metadata=sketch_metadata,
        classnames=args.classes,
    )

    all_features = torch.cat([photo_features, sketch_features], dim=0).float()
    all_metadata = photo_metadata + sketch_metadata
    coords, projection_info = compute_projection(all_features, args)
    coords = apply_layout(coords, all_metadata, args.classes, args.layout)

    points = []
    for meta, coord in zip(all_metadata, coords.tolist()):
        point = dict(meta)
        point["x"] = float(coord[0])
        point["y"] = float(coord[1])
        points.append(point)

    axis_limits = compute_axis_limits(points)
    output_dir = make_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_domain_plot(
        points=points,
        classnames=args.classes,
        domain="photo",
        output_path=output_dir / "photo_embedding.png",
        axis_limits=axis_limits,
        args=args,
    )
    save_domain_plot(
        points=points,
        classnames=args.classes,
        domain="sketch",
        output_path=output_dir / "sketch_embedding.png",
        axis_limits=axis_limits,
        args=args,
    )
    save_points_csv(points, output_dir / "embedding_points.csv")

    summary = {
        "checkpoint": str(ckpt_path),
        "photo_root": str(photo_root),
        "sketch_root": str(sketch_root),
        "classes": list(args.classes),
        "projection": projection_info,
        "layout": args.layout,
        "normalize_features": args.normalize_features,
        "frechet_distance": frechet_report,
        "num_photo_samples": len(photo_dataset),
        "num_sketch_samples": len(sketch_dataset),
        "photo_stats": photo_dataset.stats,
        "sketch_stats": sketch_dataset.stats,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "outputs": {
            "photo_embedding": str(output_dir / "photo_embedding.png"),
            "sketch_embedding": str(output_dir / "sketch_embedding.png"),
            "embedding_points": str(output_dir / "embedding_points.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Checkpoint      : {ckpt_path}")
    print(f"Photo root      : {photo_root}")
    print(f"Sketch root     : {sketch_root}")
    print(f"Classes         : {args.classes}")
    print(f"Projection      : {projection_info}")
    print(f"Layout          : {args.layout}")
    print(f"Normalize feat  : {args.normalize_features}")
    print(f"Frechet overall : {frechet_report['overall']:.6f}")
    print(f"Photo samples   : {len(photo_dataset)}")
    print(f"Sketch samples  : {len(sketch_dataset)}")
    print(f"Saved photo     : {output_dir / 'photo_embedding.png'}")
    print(f"Saved sketch    : {output_dir / 'sketch_embedding.png'}")
    print(f"Saved CSV       : {output_dir / 'embedding_points.csv'}")
    print(f"Saved summary   : {output_dir / 'summary.json'}")
    for classname, class_report in frechet_report["per_class"].items():
        print(f"Frechet[{classname}] : {class_report['distance']:.6f}")
    if missing_keys:
        print(f"Missing keys    : {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys : {unexpected_keys}")


if __name__ == "__main__":
    main()
