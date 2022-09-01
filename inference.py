# -*- coding: utf-8 -*-
"""
This script is in charge of running inference on a trained model.
The outputs (confusion matrix, box plots, histograms and UOC) are saved in a folder in the corresponding output path

@author: David Serrano Lozano, @davidserra9
"""
from os.path import join
from torch.utils.data import DataLoader
from utils.NN_utils import initialize_model
from utils.UW_dataset import UWDataset
from utils.config_parser import load_yml
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

    inference_fn(model=model,
                 loader=test_loader,
                 output_root=cfg.output_path,
                 list_classes=cfg.species,
                 mc_samples=50,
                 device=cfg.device,
                 cm=True,
                 uncertainty=True,
                 save=False,
                 load=False)
