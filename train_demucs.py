import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from einops import rearrange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data.my_dataset import MSRDataset
from demucs.demucs import Demucs
from demucs.htdemucs import HTDemucs

from data.dataset import InfiniteSampler
from models import MelRNN, MelRoFormer, UNet
from losses.gan_loss import GeneratorLoss, DiscriminatorLoss, FeatureMatchingLoss
from losses.reconstruction_loss import MultiMelSpecReconstructionLoss

from modules.discriminator.MultiPeriodDiscriminator import MultiPeriodDiscriminator
from modules.discriminator.MultiScaleDiscriminator import MultiScaleDiscriminator
from modules.discriminator.MultiFrequencyDiscriminator import MultiFrequencyDiscriminator
from modules.discriminator.MultiResolutionDiscriminator import MultiResolutionDiscriminator


# --- 辅助函数，用于修复YAML加载问题 ---
def fix_param_types(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively traverses a dictionary and converts string values that represent
    numbers (int, float, scientific notation) into their correct types.
    """
    if not isinstance(params, dict):
        return params

    for key, value in params.items():
        if isinstance(value, dict):
            fix_param_types(value)
        elif isinstance(value, str):
            try:
                params[key] = float(value)
            except (ValueError, TypeError):
                pass
    return params


class CombinedDiscriminator(nn.Module):
    """A wrapper to combine multiple discriminators into a single module."""

    def __init__(self, discriminators_config: List[Dict[str, Any]], sample_rate: int):
        super().__init__()
        disc_list = []
        for config in discriminators_config:
            name = config['name']
            params = config.get('params', {})

            if 'sample_rate' not in params:
                params['sample_rate'] = sample_rate
            if 'nch' not in params:
                params['nch'] = 1

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

    def __init__(self, config: Dict[str, Any], top_level_config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.top_level_config = top_level_config
        self.train_dataset = None
        self.valid_dataset = None

    def setup(self, stage: Optional[str] = None):
        sample_rate = self.config.get('sample_rate')
        if not isinstance(sample_rate, (int, float)):
            sample_rate = self.top_level_config.get('sample_rate')

        train_path = self.config['train_dataset']['path']
        self.train_dataset = MSRDataset(data_root=train_path, sample_rate=sample_rate)

        if 'valid_dataset' in self.config and self.config['valid_dataset'].get('path'):
            valid_path = self.config['valid_dataset']['path']
            self.valid_dataset = MSRDataset(data_root=valid_path, sample_rate=sample_rate)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **self.config['dataloader_params']
        )

    def val_dataloader(self):
        if self.valid_dataset:
            return DataLoader(
                self.valid_dataset,
                shuffle=False,
                **self.config['dataloader_params']
            )
        return None


class MusicRestorationModule(pl.LightningModule):
    """
    PyTorch Lightning module for music source restoration.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        self.automatic_optimization = False

        self.generator = self._init_generator()

        self.discriminator = CombinedDiscriminator(
            self.hparams.discriminators,
            sample_rate=self.hparams.sample_rate
        )

        loss_cfg = self.hparams.losses
        self.loss_gen_adv = GeneratorLoss(gan_type=loss_cfg.get('gan_type', 'lsgan'))
        self.loss_disc_adv = DiscriminatorLoss(gan_type=loss_cfg.get('gan_type', 'lsgan'))
        self.loss_feat = FeatureMatchingLoss()

        recon_loss_params = fix_param_types(loss_cfg.get('reconstruction_loss', {}))
        if not isinstance(recon_loss_params.get('sample_rate'), (int, float)):
            recon_loss_params['sample_rate'] = self.hparams.sample_rate
        self.loss_recon = MultiMelSpecReconstructionLoss(**recon_loss_params)

    def _init_generator(self):
        model_cfg = self.hparams.model
        params = fix_param_types(model_cfg.get('params', {}))

        if model_cfg['name'] == 'MelRNN':
            return MelRNN.MelRNN(**params)
        elif model_cfg['name'] == 'MelRoFormer':
            return MelRoFormer.MelRoFormer(**params)
        elif model_cfg['name'] == 'MelUNet':
            return UNet.MelUNet(**params)
        elif model_cfg['name'] == 'Demucs':
            return Demucs(**params)
        elif model_cfg['name'] == 'HTDemucs':
            return HTDemucs(**params)
        else:
            raise ValueError(f"Unknown model name: {model_cfg['name']}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def training_step(self, batch: tuple, batch_idx: int):
        opt_g, opt_d = self.optimizers()

        mixture, targets = batch

        # --- 手动实现梯度累积 ---
        accumulate_grad_batches = self.trainer.accumulate_grad_batches
        is_last_batch_for_accum = (batch_idx + 1) % accumulate_grad_batches == 0

        # --- Train Discriminator ---
        generated = self(mixture)

        targets_for_disc = rearrange(targets, 'b s c t -> (b s c) t')
        generated_for_disc = rearrange(generated, 'b s c t -> (b s c) t')

        real_scores, _ = self.discriminator(targets_for_disc.unsqueeze(1))
        fake_scores, _ = self.discriminator(generated_for_disc.detach().unsqueeze(1))

        d_loss, _, _ = self.loss_disc_adv(real_scores, fake_scores)

        # 每次都计算梯度，但只有在累积了足够批次后才更新权重
        self.manual_backward(d_loss / accumulate_grad_batches)  # 缩放损失
        if is_last_batch_for_accum:
            opt_d.step()
            opt_d.zero_grad()

        self.log('train/d_loss', d_loss, prog_bar=True)

        # --- Train Generator ---
        # 仅在需要更新权重时才计算生成器的损失，以节省计算
        if is_last_batch_for_accum:
            real_scores, real_fmaps = self.discriminator(targets_for_disc.unsqueeze(1))
            fake_scores, fake_fmaps = self.discriminator(generated_for_disc.unsqueeze(1))

            generated_for_recon = rearrange(generated, 'b s c t -> (b s c) t')
            targets_for_recon = rearrange(targets, 'b s c t -> (b s c) t')
            loss_recon = self.loss_recon(generated_for_recon, targets_for_recon)

            loss_adv, _ = self.loss_gen_adv(fake_scores)
            loss_feat = self.loss_feat(real_fmaps, fake_fmaps)

            loss_cfg = self.hparams.losses
            g_loss = (
                    loss_recon * loss_cfg['lambda_recon'] +
                    loss_adv * loss_cfg['lambda_gan'] +
                    loss_feat * loss_cfg['lambda_feat']
            )

            # 每次都计算梯度，但只有在累积了足够批次后才更新权重
            self.manual_backward(g_loss / accumulate_grad_batches)  # 缩放损失
            opt_g.step()
            opt_g.zero_grad()

            # 学习率调度器也只在权重更新时步进
            sch_g, sch_d = self.lr_schedulers()
            if sch_g: sch_g.step()
            if sch_d: sch_d.step()

            self.log('train/g_loss', g_loss, prog_bar=True)
            self.log('train/loss_recon', loss_recon)
            self.log('train/loss_adv', loss_adv)
            self.log('train/loss_feat', loss_feat)

    def configure_optimizers(self):
        opt_g_cfg = self.hparams.optimizer_g
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=opt_g_cfg['lr'], betas=tuple(opt_g_cfg['betas']))

        opt_d_cfg = self.hparams.optimizer_d
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=opt_d_cfg['lr'], betas=tuple(opt_d_cfg['betas']))

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

    data_module = MusicRestorationDataModule(config['data'], top_level_config=config)
    model_module = MusicRestorationModule(config)

    exp_name = f"{config['model']['name']}"
    exp_name = exp_name.replace(" ", "_")
    save_dir = Path(config['trainer']['save_dir']) / config['project_name'] / exp_name

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename="{step:08d}",
        every_n_train_steps=config['trainer']['checkpoint_save_interval'],
        save_top_k=-1,
        auto_insert_metric_name=False
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = TensorBoardLogger(
        save_dir=config['trainer']['save_dir'],
        name=config['project_name'],
        version=exp_name
    )


    trainer_config = config['trainer']
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        max_steps=trainer_config['max_steps'],
        log_every_n_steps=trainer_config['log_every_n_steps'],
        devices=trainer_config['devices'],
        precision=trainer_config['precision'],
        accelerator="gpu",
    )

    trainer.fit(model_module, datamodule=data_module)


if __name__ == '__main__':
    main()
