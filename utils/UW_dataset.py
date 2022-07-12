# -*- coding: utf-8 -*-
"""
This module contains the custom dataset class
@author: David Serrano Lozano, @davidserra9
"""
import cv2
import json
import torch
import random
import pandas as pd
import collections
import albumentations as A
from glob import glob
from os.path import join

from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as transforms
from utils.config_parser import load_yml
from utils.NN_utils import get_training_augmentations, get_validation_augmentations

class UWDataset(Dataset):
    """ Custom dataset class for loading images and labels from a list of directories divided in splits """

    def __init__(self, split_list, list_classes, train, img_per_annot=3):
        """ Initialize the dataset object. To initialize it, the function creates a list of dictionaries with the path
            of the image and its label. This object implements oversampling and data augmentation if the the set is for
            training. However, if the set is for testing, the function only loads the images and labels.

            Parameters
            ----------
            split_list : list
                List of folder paths in which the images are.
            list_classes : list
                names of the classes.
            img_per_annot : int
                Number of images per annotation. 1, 3(+-0.5s) or 5(+-1s and +-0.5s).
            train : bool
                If True, the dataset is for training. If False, the dataset is for testing. When train is True, the
                dataset is oversampled and data augmentation is applied.
        """

        self.annotations = []
        self.transforms = None
        self.train = train
        self.list_classes = list_classes

        annot = {}
        # Find the excels on the split paths
        for split in split_list:
            # Obtain all the annotation files inside the split
            annotation_files = glob(join(split, "*.ods"))

            # If there are more than one Excel file, throw an exception
            try:
                if len(annotation_files) > 1:
                    raise Exception("There are more than one annotation excel on the folder:\n{}".format(split))
                if len(annotation_files) == 0:
                    raise Exception("There are no annotation excel on the folder:\n{}".format(split))
            except Exception as e:
                print(e)
                exit()

            # Read the Excel file and create a dictionary with the labels as keys and a list of images and one-hot
            # labels as values
            df = pd.read_csv(annotation_files[0])
            for index, row in df.iterrows():
                label = row['annotation']

                # if row[list_classes].to_list().count(1) == 1:
                if label not in annot:
                    annot[label] = []
                annot[label].append({'image_root': join(split, f"{row['id_rov']:02d}_{row['img_id']:04d}"),
                                     'one-hot': row[list_classes].to_list()})

        if train:
            # Implement the oversampling, repeat the less-majority classes
            max_repeats = max([len(annot_list) for annot_list in annot.values()])

            for label, annot_list in annot.items():
                random.shuffle(annot_list)
                rep = max_repeats // len(annot_list)
                rem = max_repeats % len(annot_list)

                for i in range(rep):
                    for annot_dict in annot_list:
                        if img_per_annot == 1:
                            self.annotations.append({'image_path': annot_dict['image_root'] + "_c.jpg",
                                                     'label': label,
                                                     'one-hot': annot_dict['one-hot']})

                        elif img_per_annot == 3:
                            same_annot = sorted(glob(annot_dict['image_root'] + "*"))
                            for path in same_annot[1:4]:
                                self.annotations.append({'image_path': path,
                                                         'label': label,
                                                         'one-hot': annot_dict['one-hot']})

                        elif img_per_annot == 5:
                            same_annot = sorted(glob(annot_dict['image_root'] + "*"))
                            for path in same_annot:
                                self.annotations.append({'image_path': path,
                                                         'label': label,
                                                         'one-hot': annot_dict['one-hot']})

                        else:
                            raise Exception("img_per_annot must be 1, 3 or 5")

                for annot_dict in annot_list[:rem]:
                    if img_per_annot == 1:
                        self.annotations.append({'image_path': annot_dict['image_root'] + "_c.jpg",
                                                 'label': label,
                                                 'one-hot': annot_dict['one-hot']})

                    elif img_per_annot == 3:
                        same_annot = sorted(glob(annot_dict['image_root'] + "*"))
                        for path in same_annot[1:4]:
                            self.annotations.append({'image_path': path,
                                                     'label': label,
                                                     'one-hot': annot_dict['one-hot']})

                    elif img_per_annot == 5:
                        same_annot = sorted(glob(annot_dict['image_root'] + "*"))
                        for path in same_annot:
                            self.annotations.append({'image_path': path,
                                                     'label': label,
                                                     'one-hot': annot_dict['one-hot']})

            self.transforms = get_training_augmentations()

        else:
            for label, annot_list in annot.items():
                for annot_dict in annot_list:
                    self.annotations.append({'image_path': sorted(glob(annot_dict['image_root'] + "*")),
                                             'label': label,
                                             'one-hot': annot_dict['one-hot']})

            self.transforms = get_validation_augmentations()

        # Show the number of classes and the number of images per class
        print(f'\nNumber of classes from splits: {[x.split("/")[-1] for x in split_list]}')
        print(
            f"{len(self.annotations[0]['image_path']) if type(self.annotations[0]['image_path']) == type([]) else 1} frames per annotation")
        for label in list_classes:
            print('\t', label, len([x for x in self.annotations if x['label'] == label]))

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
            label = self.list_classes.index(label)

            return img, torch.tensor(label)

        else:
            # Read images and transform them
            image_paths = self.annotations[index]['image_path']
            images = [self.transforms(image=cv2.imread(path)[:, :, ::-1])['image'] for path in image_paths]
            images = torch.stack(images)

            # Obtain the label and encode them
            label = self.annotations[index]['label']
            label = self.list_classes.index(label)

            return images, torch.tensor(label)


if __name__ == "__main__":
    cfg = load_yml(path="../config.yml")

    train_dataset = UWDataset(
        split_list=[join(cfg.excels_path, "test_images")],
        list_classes=cfg.species,
        train=False)

    x = train_dataset[0]
