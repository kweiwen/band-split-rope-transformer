import shutil
import typing as tp
import logging
import traceback

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim import Optimizer, lr_scheduler

from data import musdbDataset, moisesdbDataset, collate_fn
from model import BandSplitRNN, BSRoformer, PLModel

from utils.callbacks import ValidationProgressBar

log = logging.getLogger(__name__)

def create_dataset(select_dataset: DictConfig, cfg: DictConfig):
    musdb_dataset = musdbDataset(
        **cfg,
    )
    moisesdb_dataset = moisesdbDataset(
        **cfg,
    )

    # map dataset class names to their instances. This allows for quick retrieval of dataset instances based on their class names.
    datasets = {
        type(musdb_dataset).__name__: musdb_dataset,
        type(moisesdb_dataset).__name__: moisesdb_dataset
    }

    # prepare a temporary list to hold dataset instances that are activated in the configuration.
    temp = []
    for key, is_dataset_active in select_dataset.items():
        # if a specific dataset is marked as active (True), append its instance to the temp list.
        if is_dataset_active: temp.append(datasets.get(key))

    # concatenate the selected datasets into a single dataset if multiple are activated.
    # this provides a unified dataset interface for subsequent DataLoader operations.
    return ConcatDataset(temp)


def initialize_loaders(cfg: DictConfig) -> tp.Tuple[DataLoader, DataLoader]:
    """
    Initializes train and validation dataloaders from configuration file.
    """
    train_dataset = create_dataset(cfg.multiple_dataset, cfg.train_dataset)
    train_loader = DataLoader(
        train_dataset,
        **cfg.train_loader,
        collate_fn=collate_fn
    )

    if hasattr(cfg, 'val_dataset'):
        val_dataset = create_dataset(cfg.multiple_dataset, cfg.val_dataset)
        val_loader = DataLoader(
            val_dataset,
            **cfg.val_loader,
            collate_fn=collate_fn)
    else:
        val_loader = None
    return (
        train_loader,
        val_loader
    )

def initialize_featurizer(
        cfg: DictConfig
) -> tp.Tuple[nn.Module, nn.Module]:
    """
    Initializes direct and inverse featurizers for audio.
    """
    featurizer = instantiate(
        cfg.featurizer.direct_transform,
    )
    inv_featurizer = instantiate(
        cfg.featurizer.inverse_transform,
    )
    return featurizer, inv_featurizer


def initialize_augmentations(
        cfg: DictConfig
) -> nn.Module:
    """
    Initializes augmentations.
    """
    augs = instantiate(cfg.augmentations)
    augs = nn.Sequential(*augs.values())
    return augs


def initialize_model(
        cfg: DictConfig
) -> tp.Tuple[nn.Module, Optimizer, lr_scheduler._LRScheduler]:
    """
    Initializes model from configuration file.
    """
    # initialize model
    model = BSRoformer(
        **cfg.model
    )
    # initialize optimizer
    if hasattr(cfg, 'opt'):
        opt = instantiate(
            cfg.opt,
            params=model.parameters()
        )
    else:
        opt = None
    # initialize scheduler
    if hasattr(cfg, 'sch'):
        if hasattr(cfg.sch, '_target_'):
            # other than LambdaLR
            sch = instantiate(
                cfg.sch,
                optimizer=opt
            )
        else:
            # if LambdaLR
            lr_lambda = lambda epoch: (
                cfg.sch.alpha ** (cfg.sch.warmup_step - epoch)
                if epoch < cfg.sch.warmup_step
                else cfg.sch.gamma ** (epoch - cfg.sch.warmup_step)
            )
            sch = torch.optim.lr_scheduler.LambdaLR(
                optimizer=opt,
                lr_lambda=lr_lambda
            )
    else:
        sch = None
    return model, opt, sch


def initialize_utils(
        cfg: DictConfig
):
    # change model and logs saving directory to logging directory of hydra
    if HydraConfig.instance().cfg is not None:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        save_dir = hydra_cfg['runtime']['output_dir']
        cfg.logger.save_dir = save_dir + cfg.logger.save_dir
        if hasattr(cfg.callbacks, 'model_ckpt'):
            cfg.callbacks.model_ckpt.dirpath = save_dir + cfg.callbacks.model_ckpt.dirpath
    # delete early stopping if there is no validation dataset
    if not hasattr(cfg, 'val_dataset') and hasattr(cfg.callbacks, 'early_stop'):
        del cfg.callbacks.early_stop
    # initialize logger and callbacks
    logger = instantiate(cfg.logger)
    callbacks = list(instantiate(cfg.callbacks).values())
    callbacks.append(ValidationProgressBar())
    return logger, callbacks


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    pl.seed_everything(42, workers=True)

    log.info(OmegaConf.to_yaml(cfg))

    log.info("Initializing loaders.")
    train_loader, val_loader = initialize_loaders(cfg)

    log.info("Initializing augmentations.")
    augs = initialize_augmentations(cfg)

    log.info("Initializing model, optimizer, scheduler.")
    model, opt, sch = initialize_model(cfg)

    log.info("Initializing Lightning logger and callbacks.")
    logger, callbacks = initialize_utils(cfg)

    log.info("Initializing Lightning modules.")
    plmodel = PLModel(
        model,
        augs,
        opt, sch,
        cfg
    )
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )

    log.info("Starting training...")
    try:
        trainer.fit(
            plmodel,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=cfg.ckpt_path
        )
    except Exception as e:
        log.error(traceback.format_exc())

    log.info("Training finished!")

    if cfg.trainer.fast_dev_run:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        shutil.rmtree(hydra_cfg['runtime']['output_dir'])


if __name__ == "__main__":
    my_app()