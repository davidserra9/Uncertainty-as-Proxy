# -*- coding: utf-8 -*-
"""
This module contains the functions to run inference on a model.
@author: David Serrano Lozano, @davidserra9
"""

import os
import cv2
import sys
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from utils.MCdropout_wrapper import MCDP_model
from utils.config_parser import load_pickle, save_pickle
from utils.uncertainty_metrics import uncertainty_box_plot, uncertainty_curve
from utils.NN_utils import get_validation_augmentations

def confusion_matrix_fn(loader, model, list_classes, device) -> plt.Figure:
    """ Function to compute and save the confusion matrix

        Parameters
        ----------
        loader : object
            pytorch dataloader
        model : object
            pytorch model
        list_classes : list
            Names of the classes'
        device : str
            'cuda' or 'cpu'
    """

    model.eval()
    # Compute the confusion matrix
    loop = tqdm(loader,  # Create the tqdm bar for visualizing the progress
                desc=f'Confusion Matrix',
                leave=True)

    y_true = []  # Network ground truth
    y_pred = []  # Network predictions

    # Do not compute the gradients on evaluation
    with torch.no_grad():
        for idx, (data, targets) in enumerate(loop):
            targets = targets.to(device)

            # If the data has 5 dimensions means that there is more than one image per annotation.
            # Tensor([batch_size, num_images_per_annot, channels, height, width])
            # The images have to be forwarded separately through the network
            if len(data.shape) == 5:
                # Separate the batches and forward them through the network obtaining num_images_per_annot tensors.
                # Each position of the outputs correspond to the same annotation
                outputs = torch.stack([model(data[:, i, :, :, :].to(device)) for i in range(data.shape[1])])

                outputs = torch.mean(outputs, dim=0)

            # If the data has 4 dimensions, standard procedure
            # Tensor([batch_size, channels, height, width])
            else:
                outputs = model(data.to(device))

            _, predictions = torch.max(outputs.data, 1)  # Obtain class with max probability

            y_true = y_true + targets.tolist()  # Add the batch ground truth labels
            y_pred = y_pred + predictions.tolist()  # Add the batch predicted labels

    cm = plot_cm(y_pred, y_true, list_classes)

    return cm


def plot_cm(y_pred, y_true, list_classes) -> plt.Figure:
    """ Function to plot the confusion matrix

        Parameters
        ----------
        y_pred : list
            list of the predicted labels
        y_true : list
            list of the ground truth labels
        list_classes : list
            list of the names of the classes

        Returns
        -------
        fig : plt.Figure
            figure with the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_prob = cm / cm.sum(axis=1, keepdims=True)

    # plt.figure(figsize=(10,7))

    labels = [l.replace(' ', '\n') for l in list_classes]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(cm_prob,
                annot=True,
                fmt='.2%',
                cmap=plt.get_cmap('Greys'),
                annot_kws={"size": 10},
                yticklabels=labels,
                xticklabels=labels,
                ax=ax)

    # title = f"Confusion Matrix"
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(labelsize=10, length=0)
    # ax.set_title(title, size=18, pad=20)
    ax.set_xlabel('Predicted Values', size=14)
    ax.set_ylabel('Actual Values', size=14)

    samples = cm.flatten().tolist()
    samples = [str(s) for s in samples]
    # samples = ['' for s in samples if s=='0']
    # samples = samples.replace('0', '')

    for text_elt, additional_text in zip(ax.texts, samples):
        ax.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
                ha='center', va='top', size=10)

    return fig


def predictive_entropy(mean):
    """ Function to compute the predictive entropy of the network

        Parameters
        ----------
        mean : np.array
            mean of the MC samples with shape (I, N, C)
            I: total number of input annotations
            N: number of images per annotation
            C: number of classes

        Return
        ------
        predictive_entropy : np.array
            predictive entropy of the network with shape (I,)
            I: total number of input annotations
    """
    epsilon = sys.float_info.min
    return -np.sum(np.mean(mean, axis=1) * np.log(np.mean(mean, axis=1) + epsilon), axis=-1)


def bhattacharyya_coefficient(dropout_predictions):
    """ Function to compute the BC from two set of points.

    Parameters
    ----------
    data1
    data2

    Returns
    -------

    """

    def hist_1d(a):
        """ Compute the histogram of a 1D array with 100 bins between 0 and 1."""
        return np.histogram(a, bins=[i * 0.05 for i in range(0, 21)])[0]

    # First, obtain the two classes with the highest predictive mean along all the images from the same
    mean = np.mean(np.mean(dropout_predictions, axis=1), axis=1)  # MC samples and images/annot mean: shape (I, C)
    clss1 = mean.argsort(axis=-1)[:, -1]  # class with the highest predictive mean: shape (I,)
    clss2 = mean.argsort(axis=-1)[:, -2]  # class with the second highest predictive mean (I,)

    # Compute the histograms of the top-2 classes for each annotation (I, bins=100)
    hist1 = np.apply_along_axis(hist_1d, axis=1, arr=np.mean(dropout_predictions, axis=2)[range(len(clss1)), :, clss1])
    hist2 = np.apply_along_axis(hist_1d, axis=1, arr=np.mean(dropout_predictions, axis=2)[range(len(clss2)), :, clss2])

    # Compute the Bhattacharyya coefficient for each annotation (I,)
    return np.sum(np.sqrt(hist1 * hist2), axis=1)


def inference_fn(model, loader, output_root, list_classes, mc_samples, device,
                 cm, uncertainty, save=False, load=False) -> None:
    """ Function to compute inference on a model.
            - Compute confusion matrix from deterministic decisions
            - Estimate uncertainty from Monte-Carlo dropout method
            - Evaluate the uncertainty estimates with std, entropy and BC
            - Evaluate the model and uncertainty individual metrics with CxI and UOC

    Parameters
    ----------
    model: pytorch model
    loader: dataloader object
    output_root: str
        output path to save the predictions and outcomes
    list_classes: list
        classes names
    mc_samples: int
        number of forward passes per image to compute uncertainty
    device: str
        "cuda" or "cpu"
    cm: bool
        confusion matrix
    uncertainty: bool
        uncertainty estimations
    save: str
        If not False, path to save the output MC predictions in a pickle file.
    load: str
        If not False, path to load the output MC predictions in a pickle file. It avoids running inference.
    """
    # Create the folder in which the output files will be saved. It is created with the experiment number taking
    # into account the amount of previous experiments/folders
    os.makedirs(output_root, exist_ok=True)  # Ensure that the output folder exists
    next_num = len(os.listdir(output_root)) + 1  # Get the next folder number
    next_folder = join(output_root, f"inf{next_num:02}")  # Create the next folder name
    os.makedirs(next_folder)  # Create the next folder

    print(f"Running inference in folder inf{next_num:02}")

    # --- CONFUSION MATRIX ---
    if cm:
        cm = confusion_matrix_fn(loader=loader,
                                 model=model,
                                 list_classes=list_classes,
                                 device=device)
        cm.savefig(join(next_folder, "confusion_matrix.png"))
        plt.close()

    # --- UNCERTAINTY ESTIMATION (MC DROPOUT) ---
    if uncertainty:

        if load:
            dropout_predictions, true_y = load_pickle(path=load)
            print(f"MC dropout predictions loaded from {load}")

        else:
            # Create the model wrapper in charge of running the MC experiment by adding, if needed, and putting dropout in
            # train mode as well as stacking the predictions.
            mc_wrapper = MCDP_model(model=model,
                                    num_classes=len(list_classes),
                                    device=device,
                                    mc_samples=mc_samples)

            dropout_predictions = np.empty((0, mc_samples, next(iter(loader))[0].shape[1], len(list_classes)))
            true_y = np.array([], dtype=np.uint8)

            # Iterate over the loader and stack all the batches predictions
            for (batch, target) in tqdm(loader, desc="Uncertainty with MC Dropout"):
                batch, target = batch.to(device), target.to(device)
                outputs = mc_wrapper(batch)  # (batch_size, mc_samples, images_per_annotations, num_classes)
                dropout_predictions = np.vstack((dropout_predictions, outputs))
                true_y = np.append(true_y, target.cpu().numpy())

            if save:
                save_pickle(path=save, data=(dropout_predictions, true_y))
                print(f"MC dropout predictions saves at {save}")

        # Mean and std of the MC predictions
        mean = np.mean(dropout_predictions, axis=1)
        std = np.std(dropout_predictions, axis=1)

        pred_y = mean.max(axis=1).argmax(axis=-1)
        pred_std = std[np.arange(mean.shape[0]), mean.max(axis=2).argmax(axis=-1), mean.max(axis=1).argmax(axis=-1)]
        pred_entropy = predictive_entropy(mean)
        pred_bc = bhattacharyya_coefficient(dropout_predictions)

        fig = uncertainty_box_plot(y_true=true_y, y_pred=pred_y, std=pred_std, entropy=pred_entropy, BC=pred_bc)
        fig.savefig(join(next_folder, "uncertainty_boxplot.png"))
        plt.close()

        fig = uncertainty_curve(y_true=true_y, y_pred=pred_y, ent=pred_entropy)
        fig.savefig(join(next_folder, "uncertainty_curve.png"))
        plt.close()
