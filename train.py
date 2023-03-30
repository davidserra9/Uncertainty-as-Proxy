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
import torch
from torch.utils.data import DataLoader
from utils.NN_utils import *
from src.ICM_dataset import ICMDataset
import hydra
from omegaconf import DictConfig, OmegaConf
from src.logging import logger
from src.training import fit, get_optimizer, get_scheduler, eval_uncertainty_model
from src.models import get_model, load_model
from src.MC_wrapper import MCWrapper
import wandb

def train(cfg: DictConfig) -> None:

    # Find which device is used
    if torch.cuda.is_available() and cfg.paths.device == "cuda":
        logger.info(f'Training the model in {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        logger.warn('CAREFUL!! Training the model with CPU')

    if "wandb" in OmegaConf.to_container(cfg.paths):
        num_exp = len(wandb.Api().runs(cfg.paths.wandb.params.project))
        wandb.init(project=cfg.paths.wandb.params.project,
                   entity=cfg.paths.wandb.params.entity,
                   name=f"{cfg.training.encoder.name}_{num_exp:02}",
                   config=OmegaConf.to_container(cfg.training))

    # Obtain the name of the wandb run
    if "wandb" in OmegaConf.to_container(cfg.paths):
        wandb_run_name = wandb.run.name

    # Create the model
    model = get_model(cfg.training.encoder)
    model = model.to("cuda")

    criterion = getattr(nn, cfg.training.loss)()
    optimizer = get_optimizer(model, cfg.training.optimizer)
    scheduler = get_scheduler(optimizer, cfg.training.scheduler)

    train_dataset = ICMDataset(path=join(cfg.paths.dataset, "train"),
                               train=True,
                               oversample=cfg.training.oversample,
                               species=cfg.paths.classes)

    valid_dataset = ICMDataset(path=join(cfg.paths.dataset, "valid"),
                               train=False,
                               species=cfg.paths.classes)

    train_loader = DataLoader(train_dataset, **cfg.training.train_dataloader)

    if cfg.training.valid_dataloader.batch_size != 1:
        logger.error("The validation batch size must be 1")
        raise ValueError("The validation batch size must be 1")

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
        cfg.paths.classes,
        cfg.paths.models,
        cfg.paths.device)

    # Load the best model
    if "wandb" in OmegaConf.to_container(cfg.paths):
        model_path = join(cfg.paths.models, f"{wandb.run.name}.pth")
    else:
        model_path = join(cfg.paths.models, f"{model.name}.pth")

    load_model(model, join(cfg.paths.models, model_path))

    # Create the test dataset and dataloader
    eval_loader = ICMDataset(path=join(cfg.paths.dataset, "test"),
                             train=False,
                             species=cfg.paths.classes)

    eval_loader = DataLoader(eval_loader, **cfg.uncertainty.eval_dataloader)

    # Create the MC Wrapper
    mc_wrapper = MCWrapper(model, cfg.uncertainty.mc_samples, cfg.paths.device)

    eval_uncertainty_model(mc_wrapper,
                           eval_loader,
                           cfg.uncertainty.mc_samples,
                           cfg.uncertainty.dropout_rate,
                           len(cfg.paths.classes),
                           "wandb" in OmegaConf.to_container(cfg.paths),
                           cfg.paths.device)
@hydra.main(config_path="config", config_name="config", version_base="1.3")
def run_training(cfg: DictConfig) -> None:
    train(cfg)
if __name__ == '__main__':
    train()
