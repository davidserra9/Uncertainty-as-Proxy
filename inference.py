import torch
from torch.utils.data import DataLoader
from utils.config_parser import load_yml
from utils.UW_dataset import UWDataset
from os.path import join
import timeit
from utils.NN_functions import initialize_model, inference_saved_model

if __name__ == "__main__":
    cfg = load_yml("config.yml")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the model
    model = initialize_model(model_name=cfg.species_classification.model,
                             num_classes=len(cfg.species),
                             load_model=True,
                             balance=cfg.species_classification.balance,
                             data_aug=cfg.species_classification.data_aug,
                             model_root=cfg.model_path)
    model.to(DEVICE)

    test_dataset = UWDataset(split_list=[join(cfg.species_dataset, f"split_{idx}") for idx in cfg.species_classification.test_splits],
                                 cfg=cfg,
                                 set="test",
                                 balance="",
                                 data_aug=False)

    test_loader = DataLoader(test_dataset,
                                 batch_size=cfg.species_classification.batch_size,
                                 num_workers=cfg.species_classification.num_workers,
                                 pin_memory=True)

    print("")
    print("----------- MODEL: {} --------------".format(model.__class__.__name__))
    print("----------- INFERENCE START --------------")
    print("")

    inference_saved_model(loader=test_loader,
                          folder_path=join(cfg.species_dataset, f"split_{cfg.species_classification.test_splits[0]}"),
                          model=model,
                          list_classes=cfg.species,
                          n_images=50,
                          n_mc_samples=100,
                          output_root=cfg.output_path,
                          device=DEVICE)