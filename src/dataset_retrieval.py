import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps

unseen_classes = [
    "bat",
    "cabin",
    "cow",
    "dolphin",
    "door",
    "giraffe",
    "helicopter",
    "mouse",
    "pear",
    "raccoon",
    "rhinoceros",
    "saw",
    "scissors",
    "seagull",
    "skyscraper",
    "songbird",
    "sword",
    "tree",
    "wheelchair",
    "windmill",
    "window",
]


def resolve_categories(opts, mode='train', used_cat=None):
    categories = os.listdir(os.path.join(opts.data_dir, 'sketch'))
    if '.ipynb_checkpoints' in categories:
        categories.remove('.ipynb_checkpoints')

    if opts.data_split > 0:
        np.random.shuffle(categories)
        if used_cat is None:
            categories = categories[:int(len(categories) * opts.data_split)]
        else:
            categories = list(set(categories) - set(used_cat))
    else:
        if mode == 'train':
            categories = list(set(categories) - set(unseen_classes))
        else:
            categories = [category for category in unseen_classes if category in categories]

    return categories

class Sketchy(torch.utils.data.Dataset):

    def __init__(self, opts, transform, mode='train', used_cat=None, return_orig=False):

        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig

        self.all_categories = resolve_categories(self.opts, mode=mode, used_cat=used_cat)

        self.all_sketches_path = []
        self.all_photos_path = {}

        for category in self.all_categories:
            self.all_sketches_path.extend(glob.glob(os.path.join(self.opts.data_dir, 'sketch', category, '*.png')))
            self.all_photos_path[category] = glob.glob(os.path.join(self.opts.data_dir, 'photo', category, '*.jpg'))

    def __len__(self):
        return len(self.all_sketches_path)
        
    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]                
        category = filepath.split(os.path.sep)[-2]
        filename = os.path.basename(filepath)
        
        neg_classes = self.all_categories.copy()
        neg_classes.remove(category)

        sk_path  = filepath
        img_path = np.random.choice(self.all_photos_path[category])
        neg_path = np.random.choice(self.all_photos_path[np.random.choice(neg_classes)])

        sk_data  = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.opts.max_size, self.opts.max_size))
        img_data = ImageOps.pad(Image.open(img_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))

        sk_tensor  = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)
        
        if self.return_orig:
            return (sk_tensor, img_tensor, neg_tensor, category, filename,
                sk_data, img_data, neg_data)
        else:
            return (sk_tensor, img_tensor, neg_tensor, category, filename)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms


class SketchyEval(torch.utils.data.Dataset):

    def __init__(self, opts, transform, mode='val', domain='sketch', used_cat=None):
        self.opts = opts
        self.transform = transform
        self.domain = domain
        self.all_categories = resolve_categories(self.opts, mode=mode, used_cat=used_cat)
        self.samples = []

        if self.domain == 'sketch':
            extension = '*.png'
        elif self.domain == 'photo':
            extension = '*.jpg'
        else:
            raise ValueError('domain must be either sketch or photo')

        for category in self.all_categories:
            self.samples.extend([
                (filepath, category) for filepath in glob.glob(
                    os.path.join(self.opts.data_dir, self.domain, category, extension)
                )
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filepath, category = self.samples[index]
        filename = os.path.basename(filepath)
        data = ImageOps.pad(
            Image.open(filepath).convert('RGB'),
            size=(self.opts.max_size, self.opts.max_size)
        )
        tensor = self.transform(data)
        return tensor, category, filename


if __name__ == '__main__':
    from experiments.options import opts
    import tqdm

    dataset_transforms = Sketchy.data_transform(opts)
    dataset_train = Sketchy(opts, dataset_transforms, mode='train', return_orig=True)
    dataset_val = Sketchy(opts, dataset_transforms, mode='val', used_cat=dataset_train.all_categories, return_orig=True)

    idx = 0
    for data in tqdm.tqdm(dataset_val):
        continue
        (sk_tensor, img_tensor, neg_tensor, filename,
            sk_data, img_data, neg_data) = data

        canvas = Image.new('RGB', (224*3, 224))
        offset = 0
        for im in [sk_data, img_data, neg_data]:
            canvas.paste(im, (offset, 0))
            offset += im.size[0]
        canvas.save('output/%d.jpg'%idx)
        idx += 1
