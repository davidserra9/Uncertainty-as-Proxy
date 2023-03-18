# -*- coding: utf-8 -*-
"""
This module contains the UnderWater dataset class
@author: David Serrano Lozano, @davidserra9
"""

import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from beautifultable import BeautifulTable
from torch.utils.data import Dataset
from utils.NN_utils import get_training_augmentations, get_validation_augmentations
from utils.config_parser import load_yml

# Training annotations divided by location
TRAINING_LOCATIONS = {"Spatangus purpureus": ["Cape Palos"],
                      "Echinaster sepositus": ["Cape Palos"],
                      "Cerianthus membranaceus": ["Cape Tiñoso"],
                      "Bonellia viridis": ["Cape Palos"],
                      "Scyliorhinus canicula": ["Cape Tiñoso"],
                      "Ophiura ophiura": ["Blanes Deep"],
                      "Background": ["Blanes Coast", "Cape Palos"]}

# Training annotations divided by videos
TRAINING_VIDEOS = [4, 8, 18, 20, 22, 23]

class ICMDataset(Dataset):
    """ Custom dataset class for loading images and labels from a list of directories divided in splits """

    def __init__(self, path, transforms):
        self.images = []
        self.labels = []
        self.transforms = transforms
        for path in glob(join(path, "*")):
            if os.path.isdir(path):
                for path

    def __len__(self) -> int:
        """ Length of the dataset. """
        return len(self.annotations)

    def __getitem__(self, index):
        """ Get the item at the given index. """

        if self.train:
            # Read image and transform it
            img = cv2.imread(self.annotations[index]['image_path'])[:, :, ::-1]
            img = self.transforms(image=img)['image']

            # Obtain the label and encode them
            label = self.annotations[index]['label']

            return img, torch.tensor(label)

        else:
            # Read image and transform
            image = self.transforms(image=cv2.imread(self.annotations[index]['image_path'])[:, :, ::-1])['image']

            # Obtain the label and encode them
            label = self.annotations[index]['label']

            return image, torch.tensor(label)

def print_dataset_stats(clss, samples):
    """ Function to print the number of samples of each class

        Parameters
        ----------
        clss: list
            class names
        samples: list
            samples numbers
    """
    bt = BeautifulTable()
    bt.columns.header = [c.split(" ")[0] for c in clss]
    bt.rows.append(samples)
    print(bt)

if __name__ == "__main__":

    cfg = load_yml("../config.yml")

    dataset = ICMDataset(dataset_path=cfg["species_dataset"], train=True, list_classes=cfg.species, videos=True)
    img, label = dataset[0]