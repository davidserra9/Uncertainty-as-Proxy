import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from packaging import version
from src.logging import logger


def get_model(cfg):
    if version.parse(torchvision.__version__) < version.parse("0.13.0"):
        logger.error("The torchvision version must be >= 0.13.0")
        raise ValueError("The torchvision version must be >= 0.13.0")

    if "efficientnet_b" in cfg.name:
        model = getattr(models, cfg.name)(weights=cfg.params.pretrained)

        model_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(model_ftrs, cfg.params.num_classes)
        model.name = cfg.name

    elif "efficientnet_v2_" in cfg.name:
        model = getattr(models, cfg.name)(weights=cfg.params.pretrained)

        model_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(model_ftrs, cfg.params.num_classes)
        model.name = cfg.name

    elif "convnext" in cfg.name:
        model = getattr(models, cfg.name)(weights=cfg.params.pretrained)

        model_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(model_ftrs, cfg.params.num_classes)
        model.name = cfg.name

    else:
        raise ValueError("Model not implemented")

    return model

def load_model():
    #TODO: load model from checkpoint
    pass

def save_model(model, optimizer, num_epoch, acc, f1, path):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": num_epoch,
        "acc": acc,
        "f1": f1
    }

    torch.save(checkpoint, path)

