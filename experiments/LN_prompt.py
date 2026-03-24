import os
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback

from src.model_LN_prompt import Model
from src.dataset_retrieval import Sketchy
from experiments.options import opts


class SaveBestAndLastCheckpoint(Callback):
    def __init__(self, dirpath, monitor='mAP'):
        super().__init__()
        self.dirpath = dirpath
        self.monitor = monitor
        self.best_path = os.path.join(dirpath, 'best.ckpt')
        self.last_path = os.path.join(dirpath, 'last.ckpt')

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        os.makedirs(self.dirpath, exist_ok=True)

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is not None:
            current_score = current_score.item()
            if current_score >= pl_module.best_metric.item():
                trainer.save_checkpoint(self.best_path)

        trainer.save_checkpoint(self.last_path)

if __name__ == '__main__':
    dataset_transforms = Sketchy.data_transform(opts)

    train_dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
    val_dataset = Sketchy(opts, dataset_transforms, mode='val', used_cat=train_dataset.all_categories, return_orig=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)
    checkpoint_dir = os.path.join('saved_models', opts.exp_name)

    checkpoint_callback = SaveBestAndLastCheckpoint(checkpoint_dir)

    ckpt_path = os.path.join(checkpoint_dir, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    trainer = Trainer(gpus=-1,
        min_epochs=1, max_epochs=2000,
        benchmark=True,
        logger=logger,
        enable_progress_bar=False,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback]
    )

    model = Model()

    print ('beginning training...good luck...')
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
