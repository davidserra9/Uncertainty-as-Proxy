# -*- coding: utf-8 -*-
"""
This module contains the functions to compute metrics and run inference to evaluate the model.
@author: David Serrano Lozano, @davidserra9
"""

import os
import sys
import cv2
import copy
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from os.path import join
from sklearn.metrics import confusion_matrix, accuracy_score, auc
from utils.NN_utils import get_validation_augmentations
from utils.cam_utils import AM_initializer
from utils.MCDP_utils import MCDP_model
from utils.reliability import reliability_diagram


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
                cmap=plt.get_cmap('Blues'),
                annot_kws={"size": 10},
                yticklabels=labels,
                xticklabels=labels,
                ax=ax)

    title = f"Confusion Matrix"
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(labelsize=10, length=0)
    ax.set_title(title, size=18, pad=20)
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


def append_dropout(model, rate=0.2) -> None:
    """ Function to append a dropout layer after a ReLu layer

        Parameters:
        ----------
        model: object
            pytorch model to append the dropout layers
        rate: float
            dropout rate
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module)
        if name == 'layer4':
            new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=True))
            setattr(model, name, new)


def enable_dropout(model) -> None:
    """ Function to enable the dropout layers during test-time

        Parameters
        ----------
        model : object
            pytorch model to enable (train mode) the dropout layers
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def dropout_train(model) -> None:
    """ Function to add dropout if needed and put dropout layers in train mode

        Parameters
        ----------
        model : object
            pytorch model to add dropout and put dropout layers in train mode
    """

    if 'resnet' in model.name:
        append_dropout(model, rate=0.5)
        enable_dropout(model)
    elif 'efficientnet' in model.name:
        enable_dropout(model)


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

def bhattacharyya_coefficient(data1, data2):
    """ Function to compute the BC from two set of points.

    Parameters
    ----------
    data1
    data2

    Returns
    -------

    """

    hist1 = np.histogram(data1, bins=[i*0.01 for i in range(0, 101)])[0]
    hist2 = np.histogram(data2, bins=[i*0.01 for i in range(0, 101)])[0]

    sq = 0
    for bin1, bin2 in zip(hist1, hist2):
        sq += np.sqrt(bin1 * bin2)

    return sq/len(data1)

def inference_fnv2(model, loader, output_root, list_classes, mc_samples, device,
                   cm, uncertainty, cam) -> None:

    # Create the folder in which the output files will be saved. It is created with the experiment number taking
    # into account the amount of previous experiments/folders
    os.makedirs(output_root, exist_ok=True)                     # Ensure that the output folder exists
    next_folder = len(os.listdir(output_root)) + 1              # Get the next folder number
    next_folder = join(output_root, f"inf{next_folder:02}")     # Create the next folder name
    os.makedirs(next_folder)                                    # Create the next folder

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
            outputs = mc_wrapper(batch)     # (batch_size, mc_samples, images_per_annotations, num_classes)
            dropout_predictions = np.vstack((dropout_predictions, outputs))
            true_y = np.append(true_y, target.cpu().numpy())

        # Mean and std of the MC predictions
        mean = np.mean(dropout_predictions, axis=1)
        std = np.std(dropout_predictions, axis=1)

        pred_y = mean.max(axis=1).argmax(axis=-1)
        pred_std = std[np.arange(mean.shape[0]), mean.max(axis=2).argmax(axis=-1), mean.max(axis=1).argmax(axis=-1)]
        pred_entropy = predictive_entropy(mean)

def inference_fn(loader, folder_path, model, list_classes, n_images, n_mc_samples, output_root, device, cm=True,
                 uncertainty=True, cam="CAM") -> None:
    """ Function to perform inference on a model.
            - Confusion Matrix
            - Uncertainty estimation with MC Dropout and Activation Maps
            - Reliability diagram (acc vs predictive mean)

        Parameters
        ----------
        loader : object
            pytorch dataloader
        folder_path : str
            string of the folder path which contains the inference images
        model : object
            pytorch model
        list_classes : list
           name of the classes
        n_images : int
            number of samples to run the example (taken randomly)
        n_mc_samples : int
            number of monte-carlo forward passes
        output_root : str
            string of the folder/root path where the output files will be saved
            (e.g. /media/david/media/TFM/experiments)
        device : str
            'cuda' or 'cpu'
        cm : bool
            if True, the confusion matrix will be computed and saved
        uncertainty : bool
            if True, the uncertainty estimation will be computed and saved
        cam : str
            "CAM", "GradCAM", "GradCAMpp", "SmoothGradCAMpp", "ScoreCAM" or "None"
        """
    random.seed(41)

    # Create the folder in which the output files will be saved
    os.makedirs(output_root, exist_ok=True)                         # Ensure that the output folder exists
    next_folder = len(os.listdir(output_root)) + 1                  # Get the next folder number
    next_folder = join(output_root, f"inf{next_folder:02}")         # Create the next folder name
    os.makedirs(next_folder)                                        # Create the next folder

    # Chose the images to run inference on
    df = pd.read_csv(glob(join(folder_path, "*.ods"))[0])           # Load the dataframe with the images gt
    all_images = glob(join(folder_path, "*.jpg"))                   # Get all the images in the folder
    random.shuffle(all_images)                                      # Shuffle the image paths
    query_images = glob(join(folder_path, "*a.jpg"))[:n_images]     # Get the first 20 images (randomly chosen)

    # Find all the images of the same annotation
    same_annot = [[r for r in sorted(all_images) if q[:-6] in r] for q in query_images]

    transformations = get_validation_augmentations()                # Get the val/test transformations
    softmax = nn.Softmax(dim=1)                                     # Create the softmax layer

    # Copy the model to have one without the dropout layers in train mode and another with the dropout layers in eval
    model.eval()                                                    # Set the model in evaluation mode
    model_dropout = copy.deepcopy(model)                            # Create a copy of the model
    dropout_train(model_dropout)                                    # Add dropout and put dropout layers in train mode
    model_dropout.to(device)                                        # Move the model to device
    model.to(device)
    # Move the model to device
    # To compute the reliability diagram
    y_true = []                                                     # List with the ground truth labels
    y_pred = []                                                     # List with the predicted labels
    pred_var = []                                                   # List with the predictive mean values
    pred_entropy = []                                               # List with the predictive entropy values
    pred_BC = []                                                    # List with the Bhattacharyya coefficient values

    # --- CONFUSION MATRIX ---
    if cm:
        cm = confusion_matrix_fn(loader=loader,
                                 model=model,
                                 list_classes=list_classes,
                                 device=device)

        cm.savefig(join(next_folder, "confusion_matrix.png"))

    # --- UNCERTAINTY ESTIMATION AND CLASS ACTIVATION ---
    if uncertainty:
        # Create the wrapper to obtain the activation maps
        wrapper = AM_initializer(model=model, technique=cam)

        # Iterate through the annotations
        for annot_idx, list_paths in enumerate(tqdm(same_annot, desc="Uncertainty Estimation and Activation Maps")):
            same_annot_img = [cv2.imread(path)[:, :, ::-1] for path in list_paths]  # Load the images

            num_img = len(list_paths)                                               # Get the number of images/annot
            # figure, axes = plt.subplots(nrows=2, ncols=num_img, figsize=(20, 8))    # Create the figure

            # Generate the uncertainty estimates of the current annotation
            dropout_predictions = np.empty((0, num_img, len(list_classes)))         # Array to store the samples

            # Iterate through the images as many times as the number of Monte-Carlo samples
            for idx_mc in range(n_mc_samples):
                images = [transformations(image=img)["image"] for img in same_annot_img]
                images = torch.stack(images).to(device)

                with torch.no_grad():
                    outputs = softmax(model_dropout(images))

                dropout_predictions = np.vstack((dropout_predictions, outputs.cpu().numpy()[np.newaxis, :, :]))

            # Compute the uncertainty estimate of the current annotation
            mean = np.mean(dropout_predictions, axis=0)                             # shape (n_samples, n_classes)
            std = np.std(dropout_predictions, axis=0)                          # shape (n_samples, n_classes)

            # # Plot the original images and the predictive mean and variance
            # for img_idx, (img, m, s) in enumerate(zip(same_annot_img, mean, std)):
            #     pred_idx = m.argmax()
            #     axes[0, img_idx].imshow(img)
            #     axes[0, img_idx].set_title(f"{list_classes[pred_idx]}\n{m[pred_idx]:.2f}-{s[pred_idx]:.4f}")
            #     axes[0, img_idx].axis('off')
            #
            #     # --- Obtain the activation map ---
            #     tensor = transformations(image=img)['image'].unsqueeze(0).to(device)
            #     cam, idx = wrapper(tensor, idx=pred_idx)
            #     heatmap = cv2.applyColorMap(
            #         cv2.resize((cam.detach().cpu().squeeze().numpy() * 255).astype(np.uint8), (img.shape[1], img.shape[0])),
            #         cv2.COLORMAP_JET)[:, :, ::-1]
            #     result = heatmap * 0.4 + img * 0.9
            #
            #     axes[1, img_idx].imshow(result / np.max(result))
            #     axes[1, img_idx].axis('off')

            img_id = list_paths[0].split("/")[-1].split("_")[1]
            y_true.append(list_classes.index(df.loc[df['img_id'] == int(img_id), 'annotation'].item()))
            i, j = np.unravel_index(mean.argmax(), mean.shape)
            y_pred.append(j)
            pred_var.append(std[i, j])
            pred_entropy.append(predictive_entropy(dropout_predictions[:, i, :]))
            # plt.hist(dropout_predictions[:, i, j], bins=[i*0.01 for i in range(0, 100)])
            # plt.hist(dropout_predictions[:, i, j+1], bins=[i*0.01 for i in range(0, 100)])
            # plt.show()

            # plt.suptitle(
            #     f"Filename: {list_paths[0].split('/')[-1]}\nGround Truth: {df.loc[df['img_id'] == int(img_id), 'annotation'].item()}\n{model.name} - {wrapper.__class__.__name__}",
            #     fontweight='bold')
            # plt.tight_layout()
            # plt.savefig(join(next_folder, f"{annot_idx}.png"))
            # plt.close()

        uncertainty_curve(y_true, y_pred, std=pred_var, entropy=pred_entropy)
        uncertainty_box_plot(y_true, y_pred, predictive_std=pred_var, predictive_entropy=pred_entropy)

        # fig = reliability_diagram(np.array(y_true), np.array(y_pred), np.array(conf), num_bins=10, draw_ece=True,
        #                           draw_bin_importance="alpha", draw_averages=True,
        #                           title="Reliability diagram", figsize=(6, 6), dpi=100,
        #                           return_fig=True)
        #
        # fig.savefig(join(next_folder, "reliability_diagram.png"))

def uncertainty_box_plot(y_true, y_pred, **metrics):
    """ Function to generate a box plot of the correct/incorrect predictions with the uncertainty estimates.

    Parameters
    ----------
    y_true : np.array
        Ground truth labels.
    y_pred : np.array
        Predicted labels.
    uncertainty : np.array
        Uncertainty estimates.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure with the box plot.
    """

    title_names = []                                # list of the names of the metrics

    # First, divide the correct/incorrect predictions into two different groups
    correct = []
    incorrect = []

    df = pd.DataFrame({'uncertainty': [val for sublist in metrics.values() for val in sublist],
                       'type': [key for key, values in metrics.items() for _ in values],
                       'correct': [len(set(i)) == 1 for i in zip(y_true, y_pred)] + [len(set(i)) == 1 for i in zip(y_true, y_pred)]})

    for name, metrics in metrics.items():
        title_names.append(name)
        bool_idx = [len(set(i)) == 1 for i in zip(y_true, y_pred)]
        correct.append(np.asarray([i for (i, v) in zip(metrics, bool_idx) if v]))
        incorrect.append(np.asarray([i for (i, v) in zip(metrics, bool_idx) if not v]))

    sns.set_theme(style="whitegrid")

    ax = sns.violinplot(y='uncertainty', x='type', hue='correct', data=df, orient="v")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2])
    plt.show()

    sns.histplot(correct[0], alpha=0.3, bins=[i*0.01 for i in range(0, 101)], label="Correct", color="green")
    sns.histplot(incorrect[0], alpha=0.3, bins=[i*0.01 for i in range(0, 101)], label="Incorrect", color="red")
    plt.legend()
    plt.xlim([0, max([max(correct[0]), max(incorrect[0])])])
    plt.show()

    print(bhattacharyya_coefficient(correct[0], incorrect[0]))

    sns.distplot(correct[0], hist=False, kde=True, color="green", label="Correct",
                 kde_kws={'shade': True, 'linewidth': 3})
    sns.distplot(incorrect[0], hist=False, kde=True, color="red", label="Incorrect",
                 kde_kws={'shade': True, 'linewidth': 3})
    plt.legend()
    plt.xlim([0, max([max(correct[0]), max(incorrect[0])])])
    plt.show()


def uncertainty_curve(y_true, y_pred, **metrics):

    for name, metric in metrics.items():
        accuracy = []
        # Sort predictions by uncertainty
        metric, y_true_ord, y_pred_ord = (list(t) for t in zip(*sorted(zip(metric, y_true, y_pred), reverse=True)))

        for idx in range(len(y_true_ord)):
            accuracy.append(accuracy_score(y_true_ord, y_pred_ord))
            y_pred_ord[idx] = y_true_ord[idx]

        au = auc(np.array(range(len(accuracy)))/len(accuracy), accuracy)
        plt.plot((np.array(range(len(accuracy)))*100)/len(accuracy), accuracy, label=f"{name} - AUC: {au:.4f}")

    plt.xlabel("Percentage of asked samples (%)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    uncertainty_box_plot(np.array([random.randint(0, 3) for _ in range(10)]),
                         np.array([random.randint(0, 3) for _ in range(10)]),
                         np.array([random.uniform(0, 1) for _ in range(10)]))
