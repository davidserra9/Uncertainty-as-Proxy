# -*- coding: utf-8 -*-
"""
This module contains the UnderWater dataset class
@author: David Serrano Lozano, @davidserra9
"""

TRAINING_LOCATIONS = {"Spatangus purpureus": ["Cape Palos"],
                      "Echinaster sepositus": ["Cape Palos"],
                      "Cerianthus membranaceus": ["Cape Tiñoso"],
                      "Bonellia viridis": ["Cape Palos"],
                      "Scyliorhinus canicula": ["Cape Tiñoso"],
                      "Ophiura ophiura": ["Blanes Deep"],
                      "Background": ["Blanes Coast", "Cape Palos"]}

TRAINING_VIDEOS = {}

import cv2
import torch
import random
import pandas as pd
from glob import glob
from os.path import join
from beautifultable import BeautifulTable
from torch.utils.data import Dataset
from utils.NN_utils import get_training_augmentations, get_validation_augmentations
from utils.config_parser import load_yml

class ICMDataset(Dataset):
    """ Custom dataset class for loading images and labels from a list of directories divided in splits """

    def __init__(self, dataset_path, list_classes, train, locations=False, videos=False):

        # Ensure correct flags
        if not locations and not videos:
            raise ValueError("At least one of the two parameters must be True")
        elif locations and videos:
            raise ValueError("Only one of the parameters locations and videos must be True")

        random.seed(42)     # Set the seed for reproducibility
        self.annotations = []   # Initialize the annotations list
        self.transforms = get_training_augmentations() if train else get_validation_augmentations()  # transformations
        self.train = train  # Set the train flag
        self.list_classes = list_classes  # Set the list of classes flag
        self.locations = locations  # Divide the dataset by locations
        self.videos = videos  # Divide the dataset by videos

        df = pd.read_csv(join(dataset_path, "dataset.csv"))  # Read the csv file

        df_concat = []
        # First, filter by species and, then, by location
        if locations and train:
            for species, locations in TRAINING_LOCATIONS.items():
                df_temp = df[df['annotation'] == species]
                df_concat.append(df_temp[df_temp['location'].isin(locations)])
            df = pd.concat(df_concat)

        elif locations and not train:
            for species, locations in TRAINING_LOCATIONS.items():
                df_temp = df[df['annotation'] == species]
                df_concat.append(df_temp[~df_temp['location'].isin(locations)])
            df = pd.concat(df_concat)

        elif videos and train:
            print("TODO")

        elif videos and not train:
            print("TODO")

        else:
            raise ValueError("Wrong combination of parameters")

        df["annotation"].drop_duplicates().sort_values()  # Find the different species

        # Get all the images classified by species
        annot = {species: [] for species in df["annotation"].drop_duplicates().sort_values()}
        for key in annot.keys():
            annot[key] = [f"{row['id_rov']:02}_{int(row['img_id']):04}.jpg" for _, row in df.loc[df['annotation'] == key].iterrows()]

        # Perform over-sampling for the training set
        if train:
            max_rep = max([len(value) for value in annot.values()])

            for label, annot_list in annot.items():
                random.shuffle(annot_list)  # Shuffle the list of image paths
                rep = max_rep // len(annot_list)  # Calculate the integer number of repetitions
                rem = max_rep % len(annot_list)  # Calculate the remaining number of repetitions

                for i in range(rep):
                    self.annotations += [{'image_path': join(dataset_path, image_path),
                                          'label': list_classes.index(label)} for image_path in annot_list]

                self.annotations += [{'image_path': join(dataset_path, image_path),
                                      'label': list_classes.index(label)} for image_path in annot_list[:rem]]

        else:
            for label, annot_list in annot.items():
                self.annotations += [{'image_path': join(dataset_path, image_path),
                                      'label': list_classes.index(label)} for image_path in annot_list]

        # Print the final number of images for each class
        print_dataset_stats(clss=list_classes,
                            samples=[len([1 for dic in self.annotations if dic['label'] == i]) for i, _ in enumerate(list_classes)])

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

    dataset = ICMDataset(dataset_path=cfg["species_dataset"], train=True, list_classes=cfg.species, locations=True)
    img, label = dataset[0]