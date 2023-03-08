import numpy as np
from torch.utils.data import DataLoader
from utils.NN_utils import initialize_model
from utils.ICM_dataset import ICMDataset
from utils.config_parser import load_yml
from utils.MCdropout_wrapper import MCDP_model
from utils.inference_utils import predictive_entropy, bhattacharyya_coefficient
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    cfg = load_yml("config.yml")

    # Initialize the model
    model = initialize_model(model_name=cfg.model,
                             num_classes=len(cfg.species),
                             load_model=True,
                             model_root=cfg.model_path)

    model.to(cfg.device)

    # Initialize datasets
    val_dataset = ICMDataset(dataset_path=cfg.icm_dataset_path,
                             list_classes=cfg.species,
                             train=False,
                             videos=True,
                             remove_multiple=False,
                             val_percentage=(1-cfg.train_percentage))

    val_loader = DataLoader(val_dataset,
                            shuffle=True,
                            batch_size=cfg.batch_size,
                            num_workers=cfg.num_workers,
                            pin_memory=True)

    print("")
    print("----------- MODEL: {} --------------".format(model.name))
    print("----------- VALIDATION START --------------")
    print("")

    dropout_predictions = np.empty((0, cfg.mc_samples, len(cfg.species)))
    true_y = np.array([], dtype=np.uint8)

    mc_wrapper = MCDP_model(model=model,
                            num_classes=len(cfg.species),
                            device=cfg.device,
                            mc_samples=cfg.mc_samples)

    for (batch, target) in tqdm(val_loader, desc="Uncertainty with MC Dropout"):
        batch, target = batch.to(cfg.device), target.to(cfg.device)
        outputs = mc_wrapper(batch)  # (batch_size, mc_samples, images_per_annotations, num_classes)
        dropout_predictions = np.vstack((dropout_predictions, outputs))
        true_y = np.append(true_y, target.cpu().numpy())

    mean = np.mean(dropout_predictions, axis=1)
    std = np.std(dropout_predictions, axis=1)

    pred_y = mean.argmax(axis=-1)
    pred_std = std[np.arange(mean.shape[0]), mean.argmax(axis=-1)]
    pred_entropy = predictive_entropy(mean)
    pred_bc = bhattacharyya_coefficient(dropout_predictions)


    # fn_list, fp_list, tn_list, tp_list, precision_list, recall_list, f1_list = [], [], [], [], [], [], []
    pred_wrong, pred_right, true_wrong, true_right, med = [], [], [], [], []
    for i in np.arange(0.01, max(pred_entropy), 0.01):
        # Predicted as wrong/right with respect to the entire dataset
        pred_wrong.append(len(np.where(pred_entropy > i)[0]) / len(true_y))
        pred_right.append(len(np.where(pred_entropy <= i)[0]) / len(true_y))

        # Predicted as wrong/right with respect the wrong/right predictions
        true_wrong.append(len(np.where((pred_y != true_y) & (pred_entropy > i))[0]) / len(np.where(pred_y != true_y)[0]))
        true_right.append(len(np.where((pred_y == true_y) & (pred_entropy <= i))[0]) / len(np.where(pred_y == true_y)[0]))

        med.append((true_wrong[-1] + true_right[-1]) / 2)
        # tp = np.sum((pred_y == true_y) & (pred_entropy <= i)) / np.sum(pred_y == true_y)
        # fp = np.sum((pred_y != true_y) & (pred_entropy <= i)) / np.sum(pred_y != true_y)
        # fn = np.sum((pred_y != true_y) & (pred_entropy > i)) / np.sum(pred_y != true_y)
        # tn = np.sum((pred_y == true_y) & (pred_entropy > i)) / np.sum(pred_y == true_y)
        #
        # fn_list.append(fn), fp_list.append(fp), tn_list.append(tn), tp_list.append(tp)
        # precision_list.append(tp / (tp + fp))
        # recall_list.append(tp / (tp + fn))
        # f1_list.append(2 * (precision_list[-1] * recall_list[-1]) / (precision_list[-1] + recall_list[-1]))

    plt.figure()
    plt.plot(np.arange(0.01, max(pred_entropy), 0.01), pred_wrong, label="predicted as wrong", c="r")
    plt.plot(np.arange(0.01, max(pred_entropy), 0.01), pred_right, label="predicted as right", c="g")
    plt.plot(np.arange(0.01, max(pred_entropy), 0.01), true_wrong, label="true as wrong")
    plt.plot(np.arange(0.01, max(pred_entropy), 0.01), true_right, label="true as right")
    plt.plot(np.arange(0.01, max(pred_entropy), 0.01), med, label="median")
    plt.legend()
    plt.show()

    # plt.figure()
    # plt.plot(np.arange(0, max(pred_entropy), 0.01), tp_list, label="TP")
    # plt.plot(np.arange(0, max(pred_entropy), 0.01), fp_list, label="FP")
    # plt.plot(np.arange(0, max(pred_entropy), 0.01), fn_list, label="FN")
    # plt.plot(np.arange(0, max(pred_entropy), 0.01), tn_list, label="TN")
    # plt.plot(np.arange(0, max(pred_entropy), 0.01), precision_list, label="Precision")
    # plt.plot(np.arange(0, max(pred_entropy), 0.01), recall_list, label="recall")
    # plt.plot(np.arange(0, max(pred_entropy), 0.01), f1_list, label="F1")
    # plt.legend()
    # plt.show()