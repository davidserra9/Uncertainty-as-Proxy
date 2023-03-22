import torch
import wandb
import time
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import join
from src.models import save_model
from src.metrics import compute_metrics
from src.logging import logger

def get_optimizer(model, cfg):
    if cfg.name.lower() == "sgd":
        # lr, wegith_decay, momentum
        optimizer = torch.optim.SGD(model.parameters(), **cfg.params)
        logger.info(f"Using SGD optimizer w/ {cfg.params}")
    elif cfg.name.lower() == "adam":
        # lr, weight_decay
        optimizer = torch.optim.Adam(model.parameters(), **cfg.params)
        logger.info(f"Using ADAM optimizer w/ {cfg.params}")

    else:
        logger.error("Optimizer not implemented")
        raise ValueError("Optimizer not implemented")

    return optimizer

def get_scheduler(optimizer, cfg):
    if cfg.name.lower() == "exponentiallr":
        # gamma
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **cfg.params)
        logger.info(f"Using ExponentialLR scheduler w/ {cfg.params}")

    elif cfg.name.lower() == "cosineannealinglr":
        # T_max, eta_min
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **cfg.params)
        logger.info(f"Using CosineAnnealingLR scheduler w/ {cfg.params}")

    elif cfg.name.lower() == "lambdalr":
        # lr_lambda
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: cfg.params.lr_lambda ** epoch)
        logger.info(f"Using LambdaLR scheduler w/ {cfg.params}")

    elif cfg.name.lower() == "none":
        scheduler = None
        logger.info("No scheduler used")
    else:
        logger.error(f"{cfg.name} scheduler not implemented")
        raise ValueError("Scheduler not implemented")
    return scheduler

def train_epoch(model, train_loader, criterion, optimizer, scheduler, log_step, epoch, wb_log, device):
    model.train()
    running_loss, total_samples, correct_samples = 0.0, 0, 0
    with tqdm(train_loader, unit="batch", leave=False, desc=f"TRAIN {epoch}") as pbar:
        for idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()               # Initialize the gradients
            outputs = model(inputs)             # Forward pass
            _, preds = torch.max(outputs, 1)    # Predictions

            loss = criterion(outputs, labels)   # Compute the loss
            loss.backward()                     # Backward pass
            optimizer.step()                    # Update the weights
            if scheduler is not None:
                scheduler.step()                    # Update the learning rate

            total_samples += labels.size(0)
            correct_samples += (preds == labels).sum().item()
            running_loss += loss.item()

            if idx % log_step == 0 and idx != 0:
                pbar.set_postfix({"lr": optimizer.param_groups[0]['lr'],
                                  "loss": running_loss / (idx+1),
                                  "acc": correct_samples / total_samples})

            if wb_log and idx % log_step == 0 and idx != 0:
                wandb.log({"train/loss": running_loss / (idx+1),
                           "train/acc": correct_samples / total_samples,
                           "train/lr": optimizer.param_groups[0]['lr']})

def valid_epoch(model, valid_loader, criterion, log_step, epoch, wb_log, cls_names, device):
    model.eval()
    running_loss = 0.0
    predictions, targets = np.empty(0), np.empty(0)
    with torch.no_grad():
        with tqdm(valid_loader, unit="batch", leave=False, desc=f"VALID {epoch}") as pbar:
            for idx, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.squeeze())
                outputs = torch.mean(outputs, dim=0, keepdim=True)
                _, preds = torch.max(outputs, 1)

                predictions = np.append(predictions, preds.cpu().numpy(), axis=0)
                targets = np.append(targets, labels.cpu().numpy(), axis=0)

                loss = criterion(outputs, labels)
                running_loss += loss.item()

                if idx % log_step == 0 and idx != 0:
                    pbar.set_postfix({"loss": running_loss / (idx+1)})

    f1, acc, cm = compute_metrics(targets, predictions, cls_names)
    if wb_log:
        wandb.log({"valid/loss": running_loss / (idx+1),
                   "valid/acc": acc,
                   "valid/f1": f1,
                   "valid/cm": wandb.Image(cm)})

    return acc, f1

def fit(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, wb_log, log_step, cls_names, output_path, device):
    max_acc, max_f1 = 0.0, 0.0
    for epoch in range(epochs):
        train_epoch(model, train_loader, criterion, optimizer, scheduler, log_step=log_step, epoch=epoch, wb_log=wb_log, device=device)
        acc, f1 = valid_epoch(model, valid_loader, criterion, log_step=log_step, epoch=epoch, wb_log=wb_log, cls_names=cls_names, device=device)

        msg = f" Epoch {epoch:02} | acc: {acc:.4f} - f1: {f1:.4f}"
        if f1 > max_f1:
            max_f1 = f1
            if wb_log:
                model_path = join(output_path, f"{wandb.run.name}.pth")
            else:
                model_path = join(output_path, f"{model.name}.pth")

            save_model(model, optimizer, epoch, acc, f1, model_path)
            msg += " | Model saved @ {}".format(model_path)

        logger.info(msg)

        if wb_log:
            wandb.summary["best_acc"] = max_acc
            wandb.summary["best_f1"] = max_f1





