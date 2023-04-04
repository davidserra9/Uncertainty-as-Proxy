import torch
import hydra
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join
from omegaconf import DictConfig, OmegaConf
from src.logging import logger
from src.models import get_model, load_model
from src.MC_wrapper import MCWrapper
from src.ICM_dataset import ICMDataset
from torch.utils.data import DataLoader
from src.metrics import predictive_entropy, uncertainty_box_plot, uncertainty_curve



@hydra.main(config_path="config", config_name="config", version_base="1.3")
def eval(cfg: DictConfig) -> None:

    # Find which device is used
    if torch.cuda.is_available() and cfg.paths.device == "cuda":
        logger.info(f'Training the model in {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        logger.warn('CAREFUL!! Training the model with CPU')

    test_dataset = ICMDataset(path=join(cfg.paths.dataset, "test"),
                               train=False,
                               species=cfg.paths.classes)

    test_loader = DataLoader(test_dataset, **cfg.training.valid_dataloader)

    model = get_model(cfg.training.encoder)
    load_model(model, "/home/david/Workspace/weights/efficientnet_b0_25.pth")
    model = model.to("cuda")

    mc_wrapper = MCWrapper(model, num_classes=len(cfg.paths.classes), mc_samples=25, dropout_rate=0.25, device="cuda")

    dropout_predictions = np.empty((0, next(iter(test_loader))[0].shape[1], 25, len(cfg.paths.classes)))
    true_y = np.array([], dtype=np.uint8)

    # Iterate over the loader and stack all the batches predictions
    for (batch, target) in tqdm(test_loader, desc="Uncertainty with MC Dropout"):
        batch, target = batch.to("cuda"), target.to("cuda")
        for b in batch:
            outputs = mc_wrapper(b)
            dropout_predictions = np.vstack((dropout_predictions, outputs[np.newaxis, :, :]))
            true_y = np.append(true_y, target.cpu().numpy())

    mean = np.mean(dropout_predictions, axis=1)

    pred_y = mean.max(axis=1).argmax(axis=-1)
    pred_entropy = predictive_entropy(mean)

    fig = uncertainty_box_plot(y_true=true_y, y_pred=pred_y, entropy=pred_entropy)
    fig.savefig("boxplot.png")
    plt.close()

    fig = uncertainty_curve(y_true=true_y, y_pred=pred_y, ent=pred_entropy)
    fig.savefig("uncertainty_curve.png")
    plt.close()

if __name__ == '__main__':
    eval()
