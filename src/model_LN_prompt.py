import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torchmetrics.functional.retrieval import retrieval_average_precision
except ImportError:
    from torchmetrics.functional import retrieval_average_precision

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

from src.clip import clip
from experiments.options import opts

def freeze_model(m):
    m.requires_grad_(False)

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.opts = opts
        self.clip, _ = clip.load('ViT-B/32', device=self.device)
        self.clip.apply(freeze_all_but_bn)
        self.clip.train()

        # Prompt Engineering
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=0.2)

        self.best_metric = -1e3
        self.latest_train_loss = None
        self.train_step_losses = []
        self.val_loss_outputs = []
        self.val_sketch_outputs = []
        self.val_photo_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.clip.parameters(), 'lr': self.opts.clip_LN_lr},
            {'params': [self.sk_prompt] + [self.img_prompt], 'lr': self.opts.prompt_lr}])
        return optimizer

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            feat = self.clip.encode_image(
                data, self.img_prompt.expand(data.shape[0], -1, -1))
        else:
            feat = self.clip.encode_image(
                data, self.sk_prompt.expand(data.shape[0], -1, -1))
        return feat

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.train_step_losses.append(loss.detach())
        return loss

    def on_train_epoch_end(self):
        if len(self.train_step_losses) == 0:
            return

        train_loss = torch.stack(self.train_step_losses).mean()
        self.latest_train_loss = train_loss.item()
        self.log('train_loss', train_loss, prog_bar=False, logger=True)
        self.train_step_losses.clear()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            sk_tensor, img_tensor, neg_tensor, category = batch[:4]
            img_feat = self.forward(img_tensor, dtype='image')
            sk_feat = self.forward(sk_tensor, dtype='sketch')
            neg_feat = self.forward(neg_tensor, dtype='image')

            loss = self.loss_fn(sk_feat, img_feat, neg_feat)
            self.val_loss_outputs.append(loss.detach())
            return loss

        image_tensor, category, filename = batch
        if dataloader_idx == 1:
            feat = self.forward(image_tensor, dtype='sketch')
            self.val_sketch_outputs.append((feat.detach(), category))
        else:
            feat = self.forward(image_tensor, dtype='image')
            self.val_photo_outputs.append((feat.detach(), category))

    def on_validation_epoch_end(self):
        if self.trainer is not None and getattr(self.trainer, 'sanity_checking', False):
            self.val_loss_outputs.clear()
            self.val_sketch_outputs.clear()
            self.val_photo_outputs.clear()
            return

        query_len = len(self.val_sketch_outputs)
        gallery_len = len(self.val_photo_outputs)
        if query_len == 0 or gallery_len == 0:
            self.val_loss_outputs.clear()
            self.val_sketch_outputs.clear()
            self.val_photo_outputs.clear()
            return

        val_loss = None
        if len(self.val_loss_outputs) > 0:
            val_loss = torch.stack(self.val_loss_outputs).mean()

        query_feat_all = torch.cat([self.val_sketch_outputs[i][0] for i in range(query_len)])
        gallery_feat_all = torch.cat([self.val_photo_outputs[i][0] for i in range(gallery_len)])
        all_sketch_category = np.array(sum([list(self.val_sketch_outputs[i][1]) for i in range(query_len)], []))
        all_photo_category = np.array(sum([list(self.val_photo_outputs[i][1]) for i in range(gallery_len)], []))

        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all))
        for idx, sk_feat in enumerate(query_feat_all):
            category = all_sketch_category[idx]
            similarity = F.cosine_similarity(sk_feat.unsqueeze(0), gallery)
            target = torch.zeros(len(gallery), dtype=torch.bool)
            target[np.where(all_photo_category == category)] = True
            ap[idx] = retrieval_average_precision(similarity.cpu(), target.cpu())

        mAP = torch.mean(ap)
        if val_loss is not None:
            self.log('val_loss', val_loss, prog_bar=False, logger=True)
        self.log('mAP', mAP, prog_bar=False, logger=True)

        if self.global_step > 0:
            self.best_metric = max(self.best_metric, mAP.item())

        current_train_loss = self.latest_train_loss
        if len(self.train_step_losses) > 0:
            current_train_loss = torch.stack(self.train_step_losses).mean().item()

        epoch_summary = ['Epoch {}'.format(self.current_epoch + 1)]
        if current_train_loss is not None:
            epoch_summary.append('train_loss: {:.4f}'.format(current_train_loss))
        if val_loss is not None:
            epoch_summary.append('val_loss: {:.4f}'.format(val_loss.item()))
        epoch_summary.append('mAP: {:.4f}'.format(mAP.item()))
        epoch_summary.append('best_mAP: {:.4f}'.format(self.best_metric))
        print(' | '.join(epoch_summary))
        self.val_loss_outputs.clear()
        self.val_sketch_outputs.clear()
        self.val_photo_outputs.clear()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['best_metric'] = self.best_metric

    def on_load_checkpoint(self, checkpoint):
        self.best_metric = checkpoint.get('best_metric', self.best_metric)
