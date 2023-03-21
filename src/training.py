import torch
import wandb
import numpy as np
from tqdm import tqdm
from src.metrics import compute_metrics

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

def fit(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, wb_log, log_step, cls_names, device):
    for epoch in range(epochs):
        train_epoch(model, train_loader, criterion, optimizer, scheduler, log_step=log_step, epoch=epoch, wb_log=wb_log, device=device)
        valid_epoch(model, valid_loader, criterion, log_step=log_step, epoch=epoch, wb_log=wb_log, cls_names=cls_names, device=device)

