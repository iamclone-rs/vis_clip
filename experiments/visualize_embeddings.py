import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_retrieval import Sketchy


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sample_paths(paths, limit):
    if limit <= 0 or len(paths) <= limit:
        return paths
    indices = np.linspace(0, len(paths) - 1, num=limit, dtype=int)
    return [paths[index] for index in indices]


class FolderDataset(Dataset):
    def __init__(self, root, classnames, max_size, max_samples_per_class):
        self.max_size = max_size
        self.transform = Sketchy.data_transform(SimpleNamespace(max_size=max_size))
        self.samples = []

        for label, classname in enumerate(classnames):
            class_dir = Path(root) / classname
            paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            self.samples.extend(
                (path, label, classname)
                for path in sample_paths(paths, max_samples_per_class)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, classname = self.samples[index]
        image = ImageOps.pad(Image.open(path).convert("RGB"), size=(self.max_size, self.max_size))
        return self.transform(image), label, classname


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--n_prompts", type=int, default=3)
    parser.add_argument("--prompt_dim", type=int, default=768)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_samples_per_class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(args, device):
    argv = list(sys.argv)
    sys.argv = [argv[0], f"--n_prompts={args.n_prompts}", f"--prompt_dim={args.prompt_dim}"]
    from src.model_LN_prompt import Model
    sys.argv = argv

    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model = Model()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    if device.type == "cpu":
        model.float()
    return model


def extract_embeddings(model, dataloader, dtype, device):
    features = []
    classnames = []

    with torch.no_grad():
        for images, _, batch_classnames in dataloader:
            images = images.to(device)
            batch_features = F.normalize(model(images, dtype=dtype).float(), dim=1)
            features.append(batch_features.cpu())
            classnames.extend(batch_classnames)

    return torch.cat(features, dim=0), classnames


def plot_domain(ax, coords, classnames, all_classnames, colors, title):
    for classname in all_classnames:
        mask = np.array([name == classname for name in classnames])
        if mask.any():
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=20,
                alpha=0.9,
                c=[colors[classname]],
                edgecolors="white",
                linewidths=0.35,
                label=classname,
            )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    photo_root = Path(args.data_dir) / "photo"
    sketch_root = Path(args.data_dir) / "sketch"

    photo_dataset = FolderDataset(photo_root, args.classes, args.max_size, args.max_samples_per_class)
    sketch_dataset = FolderDataset(sketch_root, args.classes, args.max_size, args.max_samples_per_class)

    dataloader_kwargs = {
        "batch_size": args.test_batch_size,
        "num_workers": args.workers,
        "shuffle": False,
    }
    photo_loader = DataLoader(photo_dataset, **dataloader_kwargs)
    sketch_loader = DataLoader(sketch_dataset, **dataloader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)

    photo_features, photo_classnames = extract_embeddings(model, photo_loader, "image", device)
    sketch_features, sketch_classnames = extract_embeddings(model, sketch_loader, "sketch", device)

    coords = TSNE(n_components=2, random_state=args.seed).fit_transform(
        torch.cat([photo_features, sketch_features], dim=0).numpy()
    )

    photo_coords = coords[: len(photo_features)]
    sketch_coords = coords[len(photo_features) :]
    cmap = plt.get_cmap("tab20", len(args.classes))
    colors = {classname: cmap(index) for index, classname in enumerate(args.classes)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=220, sharex=True, sharey=True)
    for ax in axes:
        ax.set_facecolor("#f5f4f1")
    fig.patch.set_facecolor("#f5f4f1")

    plot_domain(axes[0], photo_coords, photo_classnames, args.classes, colors, "Photo")
    plot_domain(axes[1], sketch_coords, sketch_classnames, args.classes, colors, "Sketch")

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, loc="upper right", frameon=True, fontsize=8)

    output_path = Path(args.output_path) if args.output_path else (
        REPO_ROOT / "visualize" / "outputs" / f"{Path(args.ckpt_path).stem}_tsne.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"Checkpoint      : {args.ckpt_path}")
    print(f"Photo root      : {photo_root}")
    print(f"Sketch root     : {sketch_root}")
    print(f"Photo samples   : {len(photo_dataset)}")
    print(f"Sketch samples  : {len(sketch_dataset)}")
    print(f"Saved figure    : {output_path}")


if __name__ == "__main__":
    main()
