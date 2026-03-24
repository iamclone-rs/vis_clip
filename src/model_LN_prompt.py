import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import retrieval_average_precision
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

        # Prompt Engineering
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=0.2)

        self.register_buffer('best_metric', torch.tensor(float('-inf')))
        self.train_epoch_loss = None

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
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        if len(outputs) == 0:
            return

        train_loss = torch.stack([output['loss'].detach() for output in outputs]).mean()
        self.train_epoch_loss = train_loss.item()
        self.log('train_loss', train_loss, prog_bar=False, logger=True)

    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        return {
            'loss': loss.detach(),
            'sk_feat': sk_feat.detach(),
            'img_feat': img_feat.detach(),
            'category': category,
        }

    def validation_epoch_end(self, val_step_outputs):
        if self.trainer.sanity_checking:
            return

        Len = len(val_step_outputs)
        if Len == 0:
            return

        val_loss = torch.stack([val_step_outputs[i]['loss'] for i in range(Len)]).mean()
        query_feat_all = torch.cat([val_step_outputs[i]['sk_feat'] for i in range(Len)])
        gallery_feat_all = torch.cat([val_step_outputs[i]['img_feat'] for i in range(Len)])
        all_category = np.array(sum([list(val_step_outputs[i]['category']) for i in range(Len)], []))


        ## mAP category-level SBIR Metrics
        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all))
        for idx, sk_feat in enumerate(query_feat_all):
            category = all_category[idx]
            distance = -1*self.distance_fn(sk_feat.unsqueeze(0), gallery)
            target = torch.zeros(len(gallery), dtype=torch.bool)
            target[np.where(all_category == category)] = True
            ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())
        
        mAP = torch.mean(ap)
        self.log('val_loss', val_loss, prog_bar=False, logger=True)
        self.log('mAP', mAP)
        self.best_metric.copy_(torch.maximum(self.best_metric, mAP.detach().to(self.best_metric.device)))

        train_loss_str = 'n/a' if self.train_epoch_loss is None else f'{self.train_epoch_loss:.4f}'
        self.print(
            f'Epoch {self.current_epoch + 1}: '
            f'train_loss={train_loss_str} '
            f'val_loss={val_loss.item():.4f} '
            f'mAP={mAP.item():.4f} '
            f'best_mAP={self.best_metric.item():.4f}'
        )
