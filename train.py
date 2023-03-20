# -*- coding: utf-8 -*-
"""
This script is in charge of training the model and evaluating it on the test set with the pytorch library.
https://pytorch.org/

The implemented architectures are:
    - VGG16, from the paper:
        "Very Deep Convolutional Networks for Large-Scale Image Recognition"
        https://arxiv.org/abs/1409.1556
    - ResNets, from the paper:
        "Deep Residual Learning for Image Recognition"
        https://arxiv.org/abs/1512.03385
    - EfficientNets, from the paper:
        "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
        https://arxiv.org/abs/1905.11946
    - EfficientNetsV2, from the paper:
        "EfficientNetV2: Smaller Models and Faster Training"
        https://arxiv.org/abs/2104.00298
    - ConvNeXts, from the paper:
        "A ConvNet for the 2020s"
        https://arxiv.org/abs/2201.03545

The training uses:
    - Adam optimizer:
        "Adam: A Method for Stochastic Optimization
        https://arxiv.org/abs/1412.6980"
        https://pytorch.org/docs/stable/optim.html
    - CrossEntropyLoss
    - Grad Scaler:

@author: David Serrano Lozano, @davidserra9
"""

import timm
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.NN_utils import *
from src.ICM_dataset import ICMDataset
import hydra
from omegaconf import DictConfig, OmegaConf
from src.logging import logger
from src.training import fit
from datetime import datetime
import wandb

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:

    # Find which device is used
    if torch.cuda.is_available() and cfg.paths.device == "cuda":
        logger.info(f'Training the model in {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        logger.warn('CAREFUL!! Training the model with CPU')

    if "wandb" in OmegaConf.to_container(cfg.paths):
        wandb.init(**cfg.paths.wandb.params,
                   name=cfg.training.encoder.name + datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                   config=OmegaConf.to_container(cfg.training))

    # Create the model
    model = timm.create_model(cfg.training.encoder.name, **cfg.training.encoder.params)
    model = model.to("cuda")

    criterion = getattr(nn, cfg.training.loss)()
    optimizer = getattr(optim, cfg.training.optimizer.name)(model.parameters(), **cfg.training.optimizer.params)
    scheduler = getattr(torch.optim.lr_scheduler, cfg.training.scheduler.name)(optimizer,
                                                                               lr_lambda=lambda epoch: cfg.training.scheduler.params.lr_lambda ** epoch)

    train_dataset = ICMDataset(path=join(cfg.paths.dataset, "train"),
                               train=True,
                               oversample=cfg.training.oversample,
                               species=cfg.paths.classes)

    valid_dataset = ICMDataset(path=join(cfg.paths.dataset, "valid"),
                              train=False,
                              species=cfg.paths.classes)

    train_loader = DataLoader(train_dataset, **cfg.training.train_dataloader)
    valid_loader = DataLoader(valid_dataset, **cfg.training.valid_dataloader)

    fit(model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        scheduler,
        cfg.training.epochs,
        "wandb" in OmegaConf.to_container(cfg.paths),
        cfg.training.log_step,
        cfg.paths.device)


if __name__ == '__main__':
    train()
