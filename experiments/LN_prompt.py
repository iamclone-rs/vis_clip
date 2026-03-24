import os
import inspect
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint

from src.model_LN_prompt import Model
from src.dataset_retrieval import Sketchy, SketchyEval
from experiments.options import opts


def resolve_num_workers(requested_workers):
    if requested_workers <= 0:
        return 0

    try:
        available_workers = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_workers = os.cpu_count()

    if not available_workers:
        return requested_workers

    effective_workers = min(requested_workers, available_workers)
    if effective_workers < requested_workers:
        print(
            'capping dataloader workers from {} to {} based on available CPUs'.format(
                requested_workers, effective_workers
            )
        )
    return effective_workers

if __name__ == '__main__':
    if torch.cuda.is_available() and hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')

    dataset_transforms = Sketchy.data_transform(opts)
    effective_workers = resolve_num_workers(opts.workers)

    train_dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
    val_dataset = Sketchy(opts, dataset_transforms, mode='val', used_cat=train_dataset.all_categories, return_orig=False)
    val_sketch_dataset = SketchyEval(
        opts, dataset_transforms, mode='val', domain='sketch', used_cat=train_dataset.all_categories
    )
    val_photo_dataset = SketchyEval(
        opts, dataset_transforms, mode='val', domain='photo', used_cat=train_dataset.all_categories
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=effective_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opts.batch_size, num_workers=effective_workers)
    val_sketch_loader = DataLoader(
        dataset=val_sketch_dataset, batch_size=opts.batch_size, num_workers=effective_workers
    )
    val_photo_loader = DataLoader(
        dataset=val_photo_dataset, batch_size=opts.batch_size, num_workers=effective_workers
    )

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    checkpoint_dir = os.path.join('saved_models', opts.exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_kwargs = {
        'monitor': 'mAP',
        'dirpath': checkpoint_dir,
        'filename': 'best',
        'mode': 'max',
        'save_top_k': 1,
        'save_last': True,
    }
    checkpoint_signature = inspect.signature(ModelCheckpoint.__init__).parameters
    if 'auto_insert_metric_name' in checkpoint_signature:
        checkpoint_kwargs['auto_insert_metric_name'] = False
    if 'enable_version_counter' in checkpoint_signature:
        checkpoint_kwargs['enable_version_counter'] = False
    checkpoint_callback = ModelCheckpoint(**checkpoint_kwargs)

    ckpt_path = os.path.join(checkpoint_dir, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    trainer_kwargs = dict(
        min_epochs=1,
        max_epochs=2000,
        benchmark=True,
        logger=logger,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback]
    )
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if 'devices' in trainer_signature:
        trainer_kwargs['accelerator'] = 'auto'
        trainer_kwargs['devices'] = 'auto'
    elif 'gpus' in trainer_signature:
        trainer_kwargs['gpus'] = -1
    if 'resume_from_checkpoint' in trainer_signature and ckpt_path is not None:
        trainer_kwargs['resume_from_checkpoint'] = ckpt_path
    if 'num_sanity_val_steps' in trainer_signature:
        trainer_kwargs['num_sanity_val_steps'] = 0
    if 'enable_progress_bar' in trainer_signature:
        trainer_kwargs['enable_progress_bar'] = False
    elif 'progress_bar_refresh_rate' in trainer_signature:
        trainer_kwargs['progress_bar_refresh_rate'] = 0

    trainer = Trainer(**trainer_kwargs)

    model = Model()

    print ('beginning training...good luck...')
    fit_kwargs = {}
    fit_signature = inspect.signature(trainer.fit).parameters
    if 'ckpt_path' in fit_signature and ckpt_path is not None:
        fit_kwargs['ckpt_path'] = ckpt_path
    trainer.fit(model, train_loader, [val_loader, val_sketch_loader, val_photo_loader], **fit_kwargs)
