import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List
from einops import rearrange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data.dataset import RawStems, InfiniteSampler
from models import MelRNN, MelRoFormer, UNet
from models.SPManba import SPMamba
from models.TIGER.models import TIGER
from losses.gan_loss import GeneratorLoss, DiscriminatorLoss, FeatureMatchingLoss
from losses.reconstruction_loss import MultiMelSpecReconstructionLoss

from modules.discriminator.MultiPeriodDiscriminator import MultiPeriodDiscriminator
from modules.discriminator.MultiScaleDiscriminator import MultiScaleDiscriminator
from modules.discriminator.MultiFrequencyDiscriminator import MultiFrequencyDiscriminator
from modules.discriminator.MultiResolutionDiscriminator import MultiResolutionDiscriminator

class CombinedDiscriminator(nn.Module):
    """A wrapper to combine multiple discriminators into a single module."""    
    def __init__(self, discriminators_config: List[Dict[str, Any]]):
        super().__init__()
        disc_list = []
        for config in discriminators_config:
            name = config['name']
            params = config['params']
            if name == 'MultiPeriodDiscriminator':
                disc_list.append(MultiPeriodDiscriminator(**params))
            elif name == 'MultiScaleDiscriminator':
                disc_list.append(MultiScaleDiscriminator(**params))
            elif name == 'MultiFrequencyDiscriminator':
                disc_list.append(MultiFrequencyDiscriminator(**params))
            elif name == 'MultiResolutionDiscriminator':
                disc_list.append(MultiResolutionDiscriminator(**params))
            else:
                raise ValueError(f"Unknown discriminator type: {name}")
        self.discriminators = nn.ModuleList(disc_list)

    def forward(self, x: torch.Tensor):
        all_scores, all_fmaps = [], []
        for disc in self.discriminators:
            scores, fmaps = disc(x)
            all_scores.extend(scores)
            all_fmaps.extend(fmaps)
        return all_scores, all_fmaps

class MusicRestorationDataModule(pl.LightningDataModule):
    """Handles data loading for training."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.train_dataset = None

    def setup(self, stage: str | None = None):
        common_params = {
            "sr": self.config['sample_rate'],
            "clip_duration": self.config['clip_duration'],
        }
        self.train_dataset = RawStems(**self.config['train_dataset'], **common_params)
    
    def train_dataloader(self):
        sampler = InfiniteSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            **self.config['dataloader_params']
        )

class MusicRestorationModule(pl.LightningModule):
    """
    PyTorch Lightning module for music source restoration,
    handling model architecture, losses, optimization, and logging.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        self.automatic_optimization = False # Needed for GANs

        # 1. Generator
        self.generator = self._init_generator()

        # 2. Discriminator
        self.discriminator = CombinedDiscriminator(self.hparams.discriminators)

        # 3. Losses
        loss_cfg = self.hparams.losses
        self.loss_gen_adv = GeneratorLoss(gan_type=loss_cfg.get('gan_type', 'lsgan'))
        self.loss_disc_adv = DiscriminatorLoss(gan_type=loss_cfg.get('gan_type', 'lsgan'))
        self.loss_feat = FeatureMatchingLoss()
        self.loss_recon = MultiMelSpecReconstructionLoss(**loss_cfg['reconstruction_loss'])
        
    def _init_generator(self):
        model_cfg = self.hparams.model
        if model_cfg['name'] == 'MelRNN':
            return MelRNN.MelRNN(**model_cfg['params'])
        elif model_cfg['name'] == 'MelRoFormer':
            return MelRoFormer.MelRoFormer(**model_cfg['params'])
        elif model_cfg['name'] == 'MelUNet':
            return UNet.MelUNet(**model_cfg['params'])
        elif model_cfg['name'] == 'SPMamba':
            return SPMamba(**model_cfg['params'])
        elif model_cfg['name'] == 'TIGER':
            return TIGER(**model_cfg['params'])
        else:
            raise ValueError(f"Unknown model name: {model_cfg['name']}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        opt_g, opt_d = self.optimizers()
        
        target = batch['target']
        mixture = batch['mixture']

        # reshape both from (b, c, t) to ((b, c) t)
        target = rearrange(target, 'b c t -> (b c) t')
        mixture = rearrange(mixture, 'b c t -> (b c) t')
        
        # --- Train Discriminator ---
        generated = self(mixture)
        
        real_scores, _ = self.discriminator(target.unsqueeze(1))
        fake_scores, _ = self.discriminator(generated.detach().unsqueeze(1))
        
        d_loss, _, _ = self.loss_disc_adv(real_scores, fake_scores)
        
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.log('train/d_loss', d_loss, prog_bar=True)

        # --- Train Generator ---
        real_scores, real_fmaps = self.discriminator(target.unsqueeze(1))
        fake_scores, fake_fmaps = self.discriminator(generated.unsqueeze(1))

        # Reconstruction Loss
        loss_recon = self.loss_recon(generated, target)
        
        # Adversarial Loss
        loss_adv, _ = self.loss_gen_adv(fake_scores)
        
        # Feature Matching Loss
        loss_feat = self.loss_feat(real_fmaps, fake_fmaps)

        loss_cfg = self.hparams.losses
        g_loss = (
            loss_recon * loss_cfg['lambda_recon'] + 
            loss_adv * loss_cfg['lambda_gan'] + 
            loss_feat * loss_cfg['lambda_feat']
        )

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log('train/g_loss', g_loss, prog_bar=True)
        self.log('train/loss_recon', loss_recon)
        self.log('train/loss_adv', loss_adv)
        self.log('train/loss_feat', loss_feat)
        
        # Step schedulers
        sch_g, sch_d = self.lr_schedulers()
        if sch_g: sch_g.step()
        if sch_d: sch_d.step()

    def configure_optimizers(self):
        # Generator Optimizer
        opt_g_cfg = self.hparams.optimizer_g
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=opt_g_cfg['lr'], betas=tuple(opt_g_cfg['betas']))
        
        # Discriminator Optimizer
        opt_d_cfg = self.hparams.optimizer_d
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=opt_d_cfg['lr'], betas=tuple(opt_d_cfg['betas']))

        # Schedulers
        if 'warm_up_steps' in self.hparams.scheduler:
            warmup_steps = self.hparams.scheduler['warm_up_steps']
            lr_lambda = lambda step: min(1.0, (step + 1) / warmup_steps)
            scheduler_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
            scheduler_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda)
            return [opt_g, opt_d], [scheduler_g, scheduler_d]
        
        return [opt_g, opt_d], []

def main():
    parser = argparse.ArgumentParser(description="Train a Music Source Restoration Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(42, workers=True)

    data_module = MusicRestorationDataModule(config['data'])
    model_module = MusicRestorationModule(config)

    exp_name = f"{config['model']['name']}"
    exp_name = exp_name.replace(" ", "_")
    save_dir = Path(config['trainer']['save_dir']) / config['project_name'] / exp_name
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename="{step:08d}",
        every_n_train_steps=config['trainer']['checkpoint_save_interval'],
        save_top_k=-1,
        auto_insert_metric_name=False
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config['trainer']['save_dir'],
        name=config['project_name'],
        version=exp_name
    )
    
    # Trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        max_steps=config['trainer']['max_steps'],
        log_every_n_steps=config['trainer']['log_every_n_steps'],
        devices=config['trainer']['devices'],
        precision=config['trainer']['precision'],
        accelerator="gpu",

        # strategy="ddp_find_unused_parameters_true",
        # use_distributed_sampler=False,
    )
    
    trainer.fit(model_module, datamodule=data_module)

if __name__ == '__main__':
    main()