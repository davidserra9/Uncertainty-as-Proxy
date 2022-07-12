import torch
from torch.utils.data import DataLoader
from utils.config_parser import load_yml
from utils.UW_dataset import UWDataset
from os.path import join
from glob import glob
import timeit
from utils.NN_utils import initialize_model
from utils.inference_utils import inference_fn

if __name__ == "__main__":
    cfg = load_yml("config.yml")

    # Initialize the model
    model = initialize_model(model_name=cfg.species_classification.model,
                             num_classes=len(cfg.species),
                             load_model=True,
                             model_root=cfg.model_path)
    model.to(cfg.device)

    # Initialize datasets
    test_dataset = UWDataset(split_list=[join(cfg.species_dataset, "test_images")],
                             list_classes=cfg.species,
                             train=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.species_classification.batch_size,
                             num_workers=cfg.species_classification.num_workers,
                             pin_memory=True)

    print("")
    print("----------- MODEL: {} --------------".format(model.name))
    print("----------- INFERENCE START --------------")
    print("")

    inference_fn(loader=test_loader,
                 folder_path=join(cfg.species_dataset, "test_images"),
                 model=model,
                 list_classes=cfg.species,
                 # n_images=len(glob(join(cfg.species_dataset, "test_images", "*.jpg"))),
                 n_images=5,
                 n_mc_samples=10,
                 output_root=cfg.output_path,
                 device=cfg.device,
                 cam="CAM")
