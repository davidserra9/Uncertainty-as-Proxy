import cv2
import json
import torch
import random
import collections
import albumentations as A
from glob import glob
from os.path import join
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as transforms
from utils.config_parser import load_yml

class UWDataset(Dataset):
    """
    Custom dataset class for loading images and labels from a list of directories divided in splits.
    """

    def __init__(self, split_list, cfg, set, balance="", data_aug=False):
        """
        Initialize the dataset object. To initialize it, the function creates a list of dictionaries with the path of
        the image and its label.
        :param split_list: list, list of the split directories
        :param config: dictionary, config
        :param set: str, train or test
        :param balance: str, oversampling, undersampling or else
        :param dataaug: boolean, True or False
        """

        self.set = set          # Save the set
        self.cfg = cfg          # Save the config file
        self.annotations = []   # Annotations paths and labels
        random.seed(42)         # To have always the same dataset and emulate results

        # The parameter set is used to differ between when there is only one image per annotation or more. If the
        # set == "train" the annotations will be stored in a list of dictionaries with the image path and its label. As
        # it can be seen:
        # [{'image_path': 'path/to/image/Ceriantus_1.jpg', 'label': 'Ceriantus'},
        #  {'image_path': 'path/to/image/Ceriantus_2.jpg', 'label': 'Ceriantus'},
        #  {'image_path': 'path/to/image/Echinaster sepositus_1.jpg', 'label': 'Echinaster sepositus'},
        #  ...]
        # However, when there is more than one frame per annotations e.g. 1 second before, 1 second after and right on
        # the time instant, the parameter set must be set to "test". The annotations will be saved in a list of
        # dictionaries with a list of the image paths and the label. As follows:
        # [{'image_path': ['path/to/image/Ceriantus_1a.jpg',
        #                  'path/to/image/Ceriantus_1b.jpg',
        #                  'path/to/image/Ceriantus_1c.jpg'],
        #   'label': 'Ceriantus'},
        #  {'image_path': ['path/to/image/Ceriantus_2a.jpg',
        #                  'path/to/image/Ceriantus_2b.jpg',
        #                  'path/to/image/Ceriantus_2c.jpg'],
        #   'label': 'Ceriantus'},
        #  {'image_path': ['path/to/image/Echinaster sepositus_1a.jpg',
        #                  'path/to/image/Echinaster sepositus_1b.jpg',
        #                  'path/to/image/Echinaster sepositus_1c.jpg'],
        #   'label': 'Echinaster sepositus'},
        #   ...]

        # If balanced == undersampling, the annotations will be balanced by removing samples from the majority classes.
        # In other words, each specie will have the same number of appearances (the least appearance object). For the
        # classes with higher occurrence, the samples are taken randomly.

        # If balanced == oversampling, the annotations will be balanced by repeating the samples from the minority
        # classes.

        # If balanced is neither undersampling or undersampling the dataset will keep its proportions
        if balance == "undersampling":

            # Obtain all the labels of the corresponding images
            img_labels = [path.split('/')[-1].split('_')[0]
                          for split in split_list for path in glob(join(split, '*.jpg'))
                          if path.split('/')[-1].split('_')[0] in cfg.species]

            # Create a dictionary with key = specie and value = empty list
            repeated_labels = {}
            for label in collections.Counter(img_labels):
                repeated_labels[label] = []

            if set == "train":
                # Append all the image paths to the corresponding key (label) (full path)
                for split in split_list:
                    for path in glob(join(split, '*.jpg')):
                        if path.split('/')[-1].split('_')[0] in cfg.species:
                            repeated_labels[path.split('/')[-1].split('_')[0]].append(path)

                # Obtain the minimum number of appearances of a class
                min_repeats = min([len(paths) for paths in repeated_labels.values()])

                # Join all the annotations in the same list maintaining the undersampling proportions
                for label, path_list in repeated_labels.items():
                    random.shuffle(path_list)
                    for path in path_list[:min_repeats]:
                        self.annotations.append({'image_path': path, 'label': label})

            else:
                # Append all the image root to the corresponding key (label) (".../Bonellia viridis0000")
                for split in split_list:
                    for path in glob(join(split, '*.jpg')):
                        if (path.split('/')[-1].split('_')[0] in cfg.species) and (path.split('.')[0][:-1] not in repeated_labels[path.split('/')[-1].split('_')[0]]):
                            repeated_labels[path.split('/')[-1].split('_')[0]].append(path.split('.')[0][:-1])

                # Obtain the minimum number of appearances of a class
                min_repeats = min([len(paths) for paths in repeated_labels.values()])

                # Join all the set of images to the same list
                for label, path_list in repeated_labels.items():
                    random.shuffle(path_list)
                    for annot_root in path_list[:min_repeats]:
                        same_annot_path = sorted(glob(annot_root + '*'))
                        self.annotations.append({'image_path': same_annot_path, 'label': label})

        elif balance == "oversampling":
            # Obtain all the labels of the corresponding images
            img_labels = [path.split('/')[-1].split('_')[0]
                          for split in split_list for path in glob(join(split, '*.jpg'))
                          if path.split('/')[-1].split('_')[0] in cfg.species]

            # Create a dictionary with key = specie and value = empty list
            repeated_labels = {}
            for label in collections.Counter(img_labels):
                repeated_labels[label] = []

            # Find the number of annotations per specie.
            if set == "train":
                for split in split_list:
                    for path in glob(join(split, '*.jpg')):
                        if path.split('/')[-1].split('_')[0] in cfg.species:
                            repeated_labels[path.split('/')[-1].split('_')[0]].append(path)

                # Obtain the minimum number of appearances of a class
                max_repeats = max([len(paths) for paths in repeated_labels.values()])

                for label, path_list in repeated_labels.items():
                    random.shuffle(path_list)
                    repetitions = max_repeats // len(path_list)
                    remainder = max_repeats % len(path_list)
                    for i in range(repetitions):
                        for path in path_list:
                            self.annotations.append({'image_path': path, 'label': label})

                    for path in path_list[:remainder]:
                        self.annotations.append({'image_path': path, 'label': label})

            else:
                for split in split_list:
                    for path in glob(join(split, '*.jpg')):
                        if (path.split('/')[-1].split('_')[0] in cfg.species) and (path.split('.')[0][:-1] not in repeated_labels[path.split('/')[-1].split('_')[0]]):
                            repeated_labels[path.split('/')[-1].split('_')[0]].append(path.split('.')[0][:-1])

                # Obtain the minimum number of appearances of a class
                max_repeats = max([len(paths) for paths in repeated_labels.values()])

                for label, path_list in repeated_labels.items():
                    random.shuffle(path_list)
                    repetitions = max_repeats // len(path_list)
                    remainder = max_repeats % len(path_list)
                    for i in range(repetitions):
                        for annot_root in path_list:
                            same_annot_path = sorted(glob(annot_root + '*'))
                            self.annotations.append({'image_path': same_annot_path, 'label': label})

                    for annot_root in path_list[:remainder]:
                        same_annot_path = sorted(glob(annot_root + '*'))
                        self.annotations.append({'image_path': same_annot_path, 'label': label})

        else:
            if set == "train":
                self.annotations = [{'image_path': path, 'label': path.split('/')[-1].split('_')[0]}
                                    for split in split_list for path in glob(join(split, '*.jpg'))
                                    if path.split('/')[-1].split('_')[0] in cfg.species]

            else:
                # Obtain all the labels of the corresponding images
                img_labels = [path.split('/')[-1].split('_')[0]
                              for split in split_list for path in glob(join(split, '*.jpg'))
                              if path.split('/')[-1].split('_')[0] in cfg.species]

                # Create a dictionary with key = specie and value = empty list
                repeated_labels = {}
                for label in collections.Counter(img_labels):
                    repeated_labels[label] = []

                for split in split_list:
                    for path in glob(join(split, '*.jpg')):
                        if (path.split('/')[-1].split('_')[0] in cfg.species) and (path.split('.')[0][:-1] not in repeated_labels[path.split('/')[-1].split('_')[0]]):
                            repeated_labels[path.split('/')[-1].split('_')[0]].append(path.split('.')[0][:-1])

                for label, path_list in repeated_labels.items():
                    for annot_root in path_list:
                        same_annot_path = sorted(glob(annot_root + '*'))
                        self.annotations.append({'image_path': same_annot_path, 'label': label})

        # Show the number of classes and the number of images per class
        print(f'\nNumber of classes from splits: {[x[-1] for x in split_list]}')
        print(f"{len(self.annotations[0]['image_path']) if type(self.annotations[0]['image_path']) == type([]) else 1} frames per annotation")
        for label in cfg.species:
            print('\t', label, len([x for x in self.annotations if x['label'] == label]))

        if data_aug:
            self.transform = A.Compose([
                A.GaussNoise(p=0.2),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=5, p=0.2),
                    A.Blur(blur_limit=5, p=0.2),
                ], p=0.2),
                A.ShiftScaleRotate(shift_limit=0.025, scale_limit=0.1, rotate_limit=10, p=0.3),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(),
                ], p=0.2),
                A.Normalize(mean=[0.4493, 0.5078, 0.4237],
                            std=[0.1263, 0.1265, 0.1169]),
                A.Resize(224, 224),
                ToTensorV2(),
            ])

        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.4493, 0.5078, 0.4237],
                            std=[0.1263, 0.1265, 0.1169]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        if self.set == "train":
            # Read image and transform it
            img = cv2.imread(self.annotations[index]['image_path'])[:,:,::-1]
            img = self.transform(image=img)['image']

            # Obtain the label and encode them
            label = self.annotations[index]['label']
            label = self.cfg.species.index(label)

            return img, torch.tensor(label)

        else:
            # Read images and transform them
            image_paths = self.annotations[index]['image_path']
            images = [self.transform(image=cv2.imread(path)[:,:,::-1])['image'] for path in image_paths]
            images = torch.stack(images)

            # Obtain the label and encode them
            label = self.annotations[index]['label']
            label = self.cfg.species.index(label)

            return images, torch.tensor(label)

if __name__ == "__main__":

    cfg = load_yml(path="../config.yml")

    train_dataset = UWDataset(
        split_list=[join(cfg.species_dataset, f"split_{idx}") for idx in cfg.species_classification.train_splits],
        cfg=cfg,
        set="train",
        balance=cfg.species_classification.balance,
        data_aug=True)