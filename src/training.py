import torch
import wandb
from tqdm import tqdm

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

def valid_epoch(model, valid_loader, criterion, log_step, epoch, wb_log, device):
    model.eval()
    running_loss, total_samples, correct_samples = 0.0, 0, 0
    with tqdm(valid_loader, unit="batch", leave=False, desc=f"VALID {epoch}") as pbar:
        for idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)             # Forward pass
            _, preds = torch.max(outputs, 1)

def fit(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, wb_log, log_step, device):
    for epoch in range(epochs):
        train_epoch(model, train_loader, criterion, optimizer, scheduler, log_step=50, epoch=epoch, wb_log=wb_log, device=device)
        valid_epoch(model, valid_loader, criterion, log_step=50, epoch=epoch, wb_log=wb_log, device=device)

