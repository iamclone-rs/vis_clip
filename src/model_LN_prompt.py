import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def category_to_id(category):
    return int(hashlib.sha1(category.encode('utf-8')).hexdigest()[:16], 16)


def average_precision(scores, target):
    target = target.bool()
    num_pos = int(target.sum().item())
    if num_pos == 0:
        raise RuntimeError('Found a query with no positive gallery items while computing mAP.')

    ranked_target = target[torch.argsort(scores, descending=True)].float()
    ranks = torch.arange(1, ranked_target.numel() + 1, device=scores.device, dtype=scores.dtype)
    precision_at_k = torch.cumsum(ranked_target, dim=0) / ranks
    return (precision_at_k * ranked_target).sum() / num_pos

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
        self.train_loss_sum = None
        self.train_loss_count = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.clip.parameters(), 'lr': self.opts.clip_LN_lr},
            {'params': [self.sk_prompt] + [self.img_prompt], 'lr': self.opts.prompt_lr}])
        return optimizer

    def on_train_epoch_start(self):
        self.train_loss_sum = torch.tensor(0.0, device=self.device)
        self.train_loss_count = torch.tensor(0.0, device=self.device)

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
        batch_size = sk_tensor.shape[0]
        self.train_loss_sum += loss.detach() * batch_size
        self.train_loss_count += batch_size
        self.log('train_loss', loss.detach(), on_step=False, on_epoch=True,
            prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        batch_size = sk_tensor.shape[0]
        if isinstance(category, str):
            category = [category]
        return {
            'loss_sum': loss.detach() * batch_size,
            'count': torch.tensor(float(batch_size), device=loss.device),
            'sk_feat': sk_feat.detach(),
            'img_feat': img_feat.detach(),
            'category_id': torch.tensor(
                [category_to_id(cat) for cat in category],
                device=loss.device,
                dtype=torch.long),
        }

    def validation_epoch_end(self, val_step_outputs):
        if self.trainer.sanity_checking:
            return

        Len = len(val_step_outputs)
        if Len == 0:
            return

        val_loss_sum = torch.stack([val_step_outputs[i]['loss_sum'] for i in range(Len)]).sum()
        val_loss_count = torch.stack([val_step_outputs[i]['count'] for i in range(Len)]).sum()
        query_feat_all = torch.cat([val_step_outputs[i]['sk_feat'] for i in range(Len)])
        gallery_feat_all = torch.cat([val_step_outputs[i]['img_feat'] for i in range(Len)])
        all_category = torch.cat([val_step_outputs[i]['category_id'] for i in range(Len)])

        if self.trainer.world_size > 1:
            query_feat_all = self.all_gather(query_feat_all).reshape(-1, query_feat_all.shape[-1])
            gallery_feat_all = self.all_gather(gallery_feat_all).reshape(-1, gallery_feat_all.shape[-1])
            all_category = self.all_gather(all_category).reshape(-1)
            val_loss_sum = self.all_gather(val_loss_sum).sum()
            val_loss_count = self.all_gather(val_loss_count).sum()
            train_loss_sum = self.all_gather(self.train_loss_sum).sum()
            train_loss_count = self.all_gather(self.train_loss_count).sum()
        else:
            train_loss_sum = self.train_loss_sum
            train_loss_count = self.train_loss_count

        val_loss = val_loss_sum / val_loss_count.clamp_min(1.0)
        train_loss = train_loss_sum / train_loss_count.clamp_min(1.0)


        ## mAP category-level SBIR Metrics
        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all), device=gallery.device)
        for idx, sk_feat in enumerate(query_feat_all):
            category = all_category[idx]
            scores = F.cosine_similarity(sk_feat.unsqueeze(0), gallery, dim=1)
            target = all_category == category
            ap[idx] = average_precision(scores, target)
        
        mAP = torch.mean(ap)
        self.log('val_loss', val_loss, prog_bar=False, logger=True)
        self.log('mAP', mAP)
        self.best_metric.copy_(torch.maximum(self.best_metric, mAP.detach().to(self.best_metric.device)))
        self.print(
            f'Epoch {self.current_epoch + 1}: '
            f'train_loss={train_loss.item():.4f} '
            f'val_loss={val_loss.item():.4f} '
            f'mAP={mAP.item():.6f} '
            f'best_mAP={self.best_metric.item():.6f}'
        )
