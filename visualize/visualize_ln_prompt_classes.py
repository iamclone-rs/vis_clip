import argparse
import csv
import json
import math
import random
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "matplotlib is required for visualization. Install it with `pip install matplotlib`."
    ) from exc

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


def _fallback_retrieval_average_precision(preds, target, top_k=None):
    top_k = top_k or preds.shape[-1]
    top_indices = preds.topk(min(top_k, preds.shape[-1]), sorted=True, dim=-1)[1]
    target = target[top_indices]

    if not target.sum():
        return torch.tensor(0.0, device=preds.device)

    positions = torch.arange(
        1, len(target) + 1, device=target.device, dtype=torch.float32
    )[target > 0]
    return (
        (torch.arange(len(positions), device=positions.device, dtype=torch.float32) + 1)
        / positions
    ).mean()


def _install_import_stubs():
    try:
        import lightning.pytorch as _lightning_pl  # noqa: F401
    except ModuleNotFoundError:
        lightning_stub = types.ModuleType("lightning")
        lightning_pytorch_stub = types.ModuleType("lightning.pytorch")

        class _LightningModule(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.trainer = None
                self.global_step = 0

            def log(self, *args, **kwargs):
                return None

        lightning_pytorch_stub.LightningModule = _LightningModule
        lightning_stub.pytorch = lightning_pytorch_stub
        sys.modules["lightning"] = lightning_stub
        sys.modules["lightning.pytorch"] = lightning_pytorch_stub

    try:
        import pytorch_lightning as _pl  # noqa: F401
    except ModuleNotFoundError:
        pytorch_lightning_stub = types.ModuleType("pytorch_lightning")

        class _LightningModule(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.trainer = None
                self.global_step = 0

            def log(self, *args, **kwargs):
                return None

        pytorch_lightning_stub.LightningModule = _LightningModule
        sys.modules["pytorch_lightning"] = pytorch_lightning_stub

    try:
        from torchmetrics.functional import (  # noqa: F401
            retrieval_average_precision as _tm_retrieval_average_precision,
        )
    except ModuleNotFoundError:
        torchmetrics_stub = types.ModuleType("torchmetrics")
        torchmetrics_functional_stub = types.ModuleType("torchmetrics.functional")
        torchmetrics_functional_stub.retrieval_average_precision = (
            _fallback_retrieval_average_precision
        )
        torchmetrics_stub.functional = torchmetrics_functional_stub
        sys.modules["torchmetrics"] = torchmetrics_stub
        sys.modules["torchmetrics.functional"] = torchmetrics_functional_stub

    try:
        import ftfy as _ftfy  # noqa: F401
    except ModuleNotFoundError:
        ftfy_stub = types.ModuleType("ftfy")

        def _fix_text(text):
            return text

        ftfy_stub.fix_text = _fix_text
        sys.modules["ftfy"] = ftfy_stub


_install_import_stubs()


def import_training_modules():
    original_argv = sys.argv[:]
    sys.argv = [original_argv[0]]
    try:
        from experiments.options import opts
        from src.dataset_retrieval import Sketchy, unseen_classes
        from src.model_LN_prompt import Model
    finally:
        sys.argv = original_argv
    return opts, Sketchy, unseen_classes, Model


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


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    label: int
    classname: str
    domain: str


class ClassFolderDataset(Dataset):
    def __init__(
        self,
        root,
        classnames,
        max_size,
        domain,
        transform,
        max_samples_per_class=0,
    ):
        self.root = Path(root)
        self.classnames = list(classnames)
        self.max_size = max_size
        self.domain = domain
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
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
        description="Visualize Sketch_LVM image/sketch embeddings as separate plots."
    )
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to .ckpt or .pth file")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class names in display order. If omitted, classes are inferred from --split.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "all", "custom"],
        help="Class split to visualize when --classes is not provided.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="",
        help="Root that contains photo/ and sketch/ folders.",
    )
    parser.add_argument("--photo_root", type=str, default="", help="Path to photo/")
    parser.add_argument("--sketch_root", type=str, default="", help="Path to sketch/")
    parser.add_argument("--photo_subdir", type=str, default="photo")
    parser.add_argument("--sketch_subdir", type=str, default="sketch")
    parser.add_argument("--output_dir", type=str, default="", help="Directory for outputs")
    parser.add_argument("--n_prompts", type=int, default=3)
    parser.add_argument("--prompt_dim", type=int, default=768)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=0,
        help="Limit samples per class per domain. Use 0 for all.",
    )
    parser.add_argument(
        "--normalize_features",
        type=str2bool,
        default=True,
        help="L2-normalize embeddings before projection.",
    )
    parser.add_argument(
        "--projection",
        type=str,
        default="tsne",
        choices=["tsne", "pca"],
        help="2D projection method.",
    )
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_n_iter", type=int, default=1500)
    parser.add_argument(
        "--layout",
        type=str,
        default="sample",
        choices=["sample", "raw"],
        help="Reposition class clusters to match a cleaner display order.",
    )
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


def resolve_split_roots(args, classnames):
    photo_root = Path(args.photo_root) if args.photo_root else None
    sketch_root = Path(args.sketch_root) if args.sketch_root else None

    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
        photo_root = photo_root or (dataset_root / args.photo_subdir)
        sketch_root = sketch_root or (dataset_root / args.sketch_subdir)

    if photo_root is None and sketch_root is None:
        raise ValueError("Provide --dataset_root or at least one of --photo_root / --sketch_root.")

    if photo_root is not None:
        photo_root = normalize_split_root(photo_root, args.photo_subdir, classnames)
    if sketch_root is not None:
        sketch_root = normalize_split_root(sketch_root, args.sketch_subdir, classnames)

    if photo_root is None:
        photo_root = infer_sibling_root(sketch_root, args.sketch_subdir, args.photo_subdir)
    if sketch_root is None:
        sketch_root = infer_sibling_root(photo_root, args.photo_subdir, args.sketch_subdir)

    return photo_root, sketch_root


def list_available_classnames(root):
    root = Path(root)
    if not root.is_dir():
        return {}
    return {
        canonicalize_class_name(path.name): path.name
        for path in root.iterdir()
        if path.is_dir()
    }


def infer_classnames(args, photo_root, sketch_root, unseen_classes):
    if args.classes:
        return list(args.classes)

    photo_dirs = list_available_classnames(photo_root)
    sketch_dirs = list_available_classnames(sketch_root)
    common_keys = sorted(set(photo_dirs) & set(sketch_dirs))
    if not common_keys:
        raise RuntimeError(
            f"No common class folders found between {photo_root} and {sketch_root}."
        )

    all_classes = [photo_dirs[key] for key in common_keys]
    unseen_keys = {canonicalize_class_name(classname) for classname in unseen_classes}

    if args.split == "custom":
        raise ValueError("Provide --classes when using --split custom.")
    if args.split == "all":
        return all_classes
    if args.split == "val":
        return [
            photo_dirs[canonicalize_class_name(classname)]
            for classname in unseen_classes
            if canonicalize_class_name(classname) in photo_dirs
            and canonicalize_class_name(classname) in sketch_dirs
        ]
    if args.split == "train":
        return [classname for classname in all_classes if canonicalize_class_name(classname) not in unseen_keys]
    return all_classes


def build_model(args, runtime_opts, model_cls, device):
    runtime_opts.n_prompts = args.n_prompts
    runtime_opts.prompt_dim = args.prompt_dim
    runtime_opts.max_size = args.max_size

    model = model_cls()
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if isinstance(checkpoint, dict) and "best_metric" in checkpoint:
        model.best_metric = checkpoint["best_metric"]

    model.eval()
    model.to(device)
    if device.type == "cpu":
        model.float()
    return model, list(missing), list(unexpected)


def extract_embeddings(model, dataloader, image_type, device, normalize_features=True):
    features = []
    metadata = []

    with torch.no_grad():
        for images, labels, classnames, paths in dataloader:
            images = images.to(device, non_blocking=device.type == "cuda")
            batch_features = model.forward(images, dtype=image_type)
            if normalize_features:
                batch_features = F.normalize(batch_features, dim=-1)
            batch_features = batch_features.detach().cpu().float()
            labels = labels.detach().cpu().tolist()
            features.append(batch_features)
            metadata.extend(
                {
                    "label": int(label),
                    "class_name": classname,
                    "domain": image_type,
                    "path": path,
                }
                for label, classname, path in zip(labels, classnames, paths)
            )

    return torch.cat(features, dim=0), metadata


def compute_pca_projection(features):
    centered = features - features.mean(dim=0, keepdim=True)
    if torch.allclose(centered.abs().sum(), torch.tensor(0.0)):
        return torch.zeros((features.shape[0], 2), dtype=torch.float32), {"method": "pca"}

    q = min(8, centered.shape[0], centered.shape[1])
    q = max(2, q)
    _, _, right_vectors = torch.pca_lowrank(centered, q=q)
    coords = centered @ right_vectors[:, :2]
    if coords.shape[1] == 1:
        coords = torch.cat([coords, torch.zeros((coords.shape[0], 1))], dim=1)
    return coords[:, :2].float(), {"method": "pca"}


def compute_projection(features, args):
    if features.shape[0] <= 1:
        return torch.zeros((features.shape[0], 2), dtype=torch.float32), {"method": "identity"}

    if args.projection == "tsne":
        if TSNE is None:
            warnings.warn("scikit-learn is not installed. Falling back to PCA.")
            return compute_pca_projection(features)

        sample_count = features.shape[0]
        perplexity = max(1.0, min(args.tsne_perplexity, float(sample_count - 1)))
        coords = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            n_iter=args.tsne_n_iter,
            random_state=args.seed,
        ).fit_transform(features.numpy())
        return torch.from_numpy(coords).float(), {
            "method": "tsne",
            "perplexity": perplexity,
            "n_iter": args.tsne_n_iter,
        }

    return compute_pca_projection(features)


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


def make_output_dir(args, classnames, split_name):
    if args.output_dir:
        return Path(args.output_dir)

    class_slug = "_".join(
        canonicalize_class_name(classname).replace(" ", "-") for classname in classnames[:12]
    )
    if len(classnames) > 12:
        class_slug = f"{class_slug}_and_more"
    ckpt_stem = Path(args.ckpt_path).stem
    return REPO_ROOT / "visualize" / "outputs" / f"{ckpt_stem}_{split_name}_{class_slug}"


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    runtime_opts, sketchy_dataset_cls, unseen_classes, model_cls = import_training_modules()
    split_name = "custom" if args.classes else args.split

    provisional_classnames = list(args.classes) if args.classes else list(unseen_classes)
    photo_root, sketch_root = resolve_split_roots(args, provisional_classnames)
    classnames = infer_classnames(args, photo_root, sketch_root, unseen_classes)
    if not classnames:
        raise RuntimeError(
            f"No classes selected for split `{split_name}` under {photo_root} and {sketch_root}."
        )

    photo_root, sketch_root = resolve_split_roots(args, classnames)
    runtime_opts.max_size = args.max_size
    transform = sketchy_dataset_cls.data_transform(runtime_opts)

    photo_dataset = ClassFolderDataset(
        root=photo_root,
        classnames=classnames,
        max_size=args.max_size,
        domain="image",
        transform=transform,
        max_samples_per_class=args.max_samples_per_class,
    )
    sketch_dataset = ClassFolderDataset(
        root=sketch_root,
        classnames=classnames,
        max_size=args.max_size,
        domain="sketch",
        transform=transform,
        max_samples_per_class=args.max_samples_per_class,
    )

    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "shuffle": False,
        "pin_memory": args.device.startswith("cuda"),
    }
    photo_loader = DataLoader(photo_dataset, **dataloader_kwargs)
    sketch_loader = DataLoader(sketch_dataset, **dataloader_kwargs)

    device = torch.device(args.device)
    model, missing_keys, unexpected_keys = build_model(args, runtime_opts, model_cls, device)

    photo_features, photo_metadata = extract_embeddings(
        model=model,
        dataloader=photo_loader,
        image_type="image",
        device=device,
        normalize_features=args.normalize_features,
    )
    sketch_features, sketch_metadata = extract_embeddings(
        model=model,
        dataloader=sketch_loader,
        image_type="sketch",
        device=device,
        normalize_features=args.normalize_features,
    )

    all_features = torch.cat([photo_features, sketch_features], dim=0).float()
    all_metadata = photo_metadata + sketch_metadata
    coords, projection_info = compute_projection(all_features, args)
    coords = apply_layout(coords, all_metadata, classnames, args.layout)

    points = []
    for meta, coord in zip(all_metadata, coords.tolist()):
        point = dict(meta)
        point["x"] = float(coord[0])
        point["y"] = float(coord[1])
        point["domain"] = "photo" if point["domain"] == "image" else point["domain"]
        points.append(point)

    axis_limits = compute_axis_limits(points)
    output_dir = make_output_dir(args, classnames, split_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_domain_plot(
        points=points,
        classnames=classnames,
        domain="photo",
        output_path=output_dir / "photo_embedding.png",
        axis_limits=axis_limits,
        args=args,
    )
    save_domain_plot(
        points=points,
        classnames=classnames,
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
        "classes": classnames,
        "split": split_name,
        "projection": projection_info,
        "layout": args.layout,
        "normalize_features": args.normalize_features,
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
    print(f"Split           : {split_name}")
    print(f"Classes         : {classnames}")
    print(f"Projection      : {projection_info}")
    print(f"Layout          : {args.layout}")
    print(f"Normalize feat  : {args.normalize_features}")
    print(f"Photo samples   : {len(photo_dataset)}")
    print(f"Sketch samples  : {len(sketch_dataset)}")
    print(f"Saved photo     : {output_dir / 'photo_embedding.png'}")
    print(f"Saved sketch    : {output_dir / 'sketch_embedding.png'}")
    print(f"Saved CSV       : {output_dir / 'embedding_points.csv'}")
    print(f"Saved summary   : {output_dir / 'summary.json'}")
    if missing_keys:
        print(f"Missing keys    : {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys : {unexpected_keys}")


if __name__ == "__main__":
    main()
