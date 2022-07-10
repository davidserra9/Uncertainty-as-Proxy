import os
import sys
import albumentations as A
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import random
import torch
import torch.nn as nn
import torchvision.models as models
from os.path import join
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from glob import glob
from beautifultable import BeautifulTable
from sklearn.metrics import confusion_matrix
from albumentations.pytorch import ToTensorV2
from utils.reliability import *

def initialize_model(model_name, num_classes, load_model, model_root):
    """ Function to initialize the model depending on the desired architecture.

        Parameters
        ----------
        model_name : string
            name of the model to initialize
        num_classes : int
            number of classes to predict
        load_model : boolean
            if True, load the model from the disk
        model_root : string
            string of the folder/root path where the model will be saved

        Returns
        -------
        model : object
    """

    IMP_ARCH = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4',
                'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']

    try:
        if model_name in IMP_ARCH:
            model = getattr(models, model_name)(pretrained=True)
            if 'resnet' in model_name:
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
                model.name = model_name

            elif 'efficientnet' in model_name:
                num_ftrs = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(num_ftrs, num_classes)
                model.name = model_name

        else:
            raise ValueError('Model not implemented')

    except Exception as e:
        print(e)
        print(f'Model {model_name} not found. Please check the model name or torchvision version.')
        print(f"Models implemented:\nResNet18/34/50/101/152\nEfficientNet_b0-7")
        exit()

    # If load_model==True, load the weights of the model
    if load_model:
        # Load the desired model
        model_path = join(model_root, model_name + ".pth.tar")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded from {}'.format(model_path))

        # Print the model parameters
        bt = BeautifulTable()
        bt.columns.header = ["architecture", "epoch", "accuracy", "f1 score"]
        bt.rows.append([model_name, checkpoint['epoch'], checkpoint['test_acc'], checkpoint['f1']])
        print()
        print(bt)

    else:
        print('Model initialized with weights from ImageNet')

    return model

def save_model(model, optimizer, num_epoch, acc, f1, model_root):
    """ Function to save the model in the desired folder.

        Parameters
        ----------
        model : object
            pytorch model
        optimizer : object
            pytorch optimizer
        num_epoch : int
            number of epochs trained on
        acc : float
            accuracy of the model
        f1 : float
            f1 score of the model
        model_root : str
            string of the folder/root path where the model will be saved
    """

    # Create the checkpoint dictionary
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": num_epoch,
        "test_acc": acc,
        "f1": f1
    }

    model_path = join(model_root, model.name + ".pth.tar")
    torch.save(checkpoint, model_path)
    print("Model saved in {}".format(model_path))

def train_fn(loader, model, optimizer, loss_fn, scaler, device, epoch_num):
    """ Function to train the model with one epoch

        Parameters
        ----------
        loader : object
            pytorch dataloader
        model : object
            pytorch model
        optimizer : object
            pytorch optimizer
        loss_fn : object
            pytorch loss function
        scaler : object
            pytorch scaler
        device : string
            cuda or cpu
        epoch_num : int
            current epoch number

        Returns
        -------
        epoch_acc : float
            accuracy of the model
        epoch_loss : float
            loss of the model
    """

    model.train()

    loop = tqdm(loader,                                         # Create the tqdm bar for visualizing the progress
                desc=f'EPOCH {epoch_num} TRAIN',
                leave=True)

    correct = 0                                                 # accumulated correct predictions
    total_samples = 0                                           # accumulated total predictions
    loss_sum = 0                                                # accumulated loss

    for idx, (data, targets) in enumerate(loop):
        data, targets = data.to(device), targets.to(device)     # data and labels to device
        optimizer.zero_grad()                                   # Initialize gradients

        outputs = model(data)                                   # Forward pass
        loss = loss_fn(outputs, targets)                        # Compute the loss
        _, predictions = torch.max(outputs.data, 1)             # Obtain the classes with higher probability

        total_samples += data.size(0)                           # Subtotal of the predictions
        correct += (predictions == targets).sum().item()        # Subtotal of the correct predictions
        loss_sum += loss.item()                                 # Subtotal of the correct losses

        scaler.scale(loss).backward()                           # Backward pass
        scaler.step(optimizer)                                  # Update the weights
        scaler.update()                                         # Update the scale

        loop.set_postfix(acc=correct / total_samples,
                         loss=loss_sum / (idx+1))

    epoch_acc = correct / total_samples                         # Epoch accuracy
    epoch_loss = loss_sum / len(loader)                         # Epoch loss

    return epoch_acc, epoch_loss

def eval_fn(loader, model, loss_fn, device, epoch_num):
    """ Function to evaluate the model with one epoch

        Parameters
        ----------
        loader : object
            pytorch dataloader
        model : object
            pytorch model
        loss_fn : object
            pytorch loss function
        device : string
            cuda or cpu
        epoch_num : int
            current epoch number

        Returns
        -------
        epoch_acc : float
            accuracy of the model (current epoch)
        epoch_loss : float
            loss of the model   (current epoch)
        epoch_f1 : float
            f1 score of the model (current epoch)
    """

    model.eval()

    loop = tqdm(loader,                                         # Create the tqdm bar for visualizing the progress
                desc=f'EPOCH {epoch_num}  TEST',
                leave=True)

    correct = 0                                                 # Accumulated correct predictions
    total_samples = 0                                           # Accumulated total predictions
    loss_sum = 0                                                # Accumulated loss
    y_true = []                                                 # Network ground truth
    y_pred = []                                                 # Network predictions

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

            loss = loss_fn(outputs, targets)                     # Compute loss

            _, predictions = torch.max(outputs.data, 1)          # Obtain class with max probability

            y_true = y_true + targets.tolist()                  # Add the batch ground truth labels
            y_pred = y_pred + predictions.tolist()              # Add the batch predicted labels

            total_samples += data.size(0)  # subtotal of the predictions
            correct += (predictions == targets).sum().item()  # subtotal of the correct predictions
            loss_sum += loss.item()  # Subtotal of the correct losses

            loop.set_postfix(acc=correct / total_samples,
                             loss=loss_sum / (idx + 1),
                             f1=f1_score(y_true=y_true, y_pred=y_pred, average='macro'))

        epoch_acc = correct / total_samples
        epoch_loss = loss_sum / (idx + 1)
        epoch_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

        return epoch_acc, epoch_loss, epoch_f1

def confusion_matrix_fn(loader, model, list_classes, output_path, device) -> None:
    """ Function to compute and save the confusion matrix

        Parameters
        ----------
        loader : object
            pytorch dataloader
        model : object
            pytorch model
        list_classes : list
            list of the names of the classes
        output_path : str
            path to the folder where the confusion matrix will be saved
            (.../experiments/infxx/)
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
    cm.savefig(join(output_path, "confusion_matrix.png"))

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

def predictive_entropy(dropout_predictions):
    """ Function to compute the predictive entropy of the network

        Parameters
        ----------
        dropout_predictions : np.array
    """
    epsilon = sys.float_info.min
    predictive_entropy = -np.sum(np.mean(dropout_predictions, axis=0) * np.log(np.mean(dropout_predictions, axis=0) + epsilon), axis=-1)

    return predictive_entropy

def return_CAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255*cam_img)
    return cv2.resize(cam_img, size_upsample)

def register_CAM_hooks(model):

    feature_blobs = []
    def hook_feature(module, input, output):
        feature_blobs.append(output.data.cpu().numpy())

    if 'resnet' in model.name:
        model._modules.get('layer4').register_forward_hook(hook_feature)
    elif 'efficientnet' in model.name:
        model._modules.get('features').register_forward_hook(hook_feature)
    return feature_blobs

def inference_saved_model(loader, folder_path, model, list_classes, n_images, n_mc_samples, output_root, device) -> None:
    """ Function to perform inference on a saved model.
            - Inference of image samples
            - Confusion Matrix
            - Monte-Carlo DropOut to Uncertainty estimation

            Parameters
            ----------
            loader : object
                pytorch dataloader
            folder_path : str
                string of the folder path which contains the inference images
            model : object
                pytorch model
            list_classes : list
                list with the name of the classes
            n_images : int
                number of samples to run the example (taken randomly)
            n_mc_samples : int
                number of monte-carlo forward passes
            output_root : str
                string of the folder/root path where the output files will be saved
                (e.g. /media/david/media/TFM/experiments)
            device : str
                'cuda' or 'cpu'
        """
    random.seed(41)

    # Create the folder in which the output files will be saved
    os.makedirs(output_root, exist_ok=True)
    next_folder = len(os.listdir(output_root)) + 1
    next_folder = join(output_root, f"inf{next_folder:02}")
    os.makedirs(next_folder)

    model.eval()
    model.to(device)

    confusion_matrix_fn(loader=loader,
                        model=model,
                        list_classes=list_classes,
                        output_path=next_folder,
                        device=device)

    model.eval()                # Ensure that the model is in evaluation mode
    dropout_train(model)        # Dropout in train mode to generate the MC samples

    # Chose the images to run inference on
    df = pd.read_csv(glob(join(folder_path, "*.ods"))[0])
    all_images = glob(join(folder_path, "*.jpg"))  # Get all the images in the folder
    random.shuffle(all_images)  # Shuffle the image paths
    query_images = glob(join(folder_path, "*a.jpg"))[:n_images]  # Get the first 20 images (randomly chosen)

    # Find all the images of the same annotation
    same_annot = [[r for r in sorted(all_images) if q[:-6] in r] for q in query_images]

    transformations = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.4493, 0.5078, 0.4237],
                            std=[0.1263, 0.1265, 0.1169]),
                ToTensorV2()
            ])
    # --- Reliability diagram ---
    y_true = []
    y_pred = []
    conf = []

    # --- CAMs and MC Dropout ---
    feature_globs = register_CAM_hooks(model)
    params = list(model.parameters())
    weight_softmax = params[-2].data.cpu().numpy()
    softmax = nn.Softmax(dim=1)

    for idx_annot, list_paths in enumerate(tqdm(same_annot, desc="Inference")):
        #feature_globs = []
        same_annot_img = [cv2.imread(path)[:, :, ::-1] for path in list_paths]

        num_img = len(same_annot_img)
        figure, axes = plt.subplots(nrows=2, ncols=num_img, figsize=(20, 8))

        # --- Generate CAMs ---
        for idx, image in enumerate(same_annot_img):

            outputs = model(transformations(image=image)["image"].unsqueeze(0).to(device))
            probs = F.softmax(outputs, dim=1).data.squeeze()
            class_idx = torch.argmax(probs).item()
            CAM = return_CAM(feature_globs[-1], weight_softmax, class_idx)

            heatmap = cv2.applyColorMap(cv2.resize(CAM, (image.shape[1], image.shape[0])), cv2.COLORMAP_JET)[:, :, ::-1]
            result = heatmap * 0.4 + image * 0.9
            axes[1, idx].imshow(result / np.max(result))
            axes[1, idx].set_title(f"Pred: {list_classes[class_idx]}")
            axes[1, idx].axis('off')

        # --- Generate Uncertainty estimates ---
        dropout_predictions = np.empty((0, num_img, len(list_classes)))

        for idx_mc in range(n_mc_samples):
            images = [transformations(image=cv2.imread(img_path)[:, :, ::-1])["image"] for img_path in list_paths]
            images = torch.stack(images).to(device)

            with torch.no_grad():
                outputs = softmax(model(images))

            dropout_predictions = np.vstack((dropout_predictions, outputs.cpu().numpy()[np.newaxis, :, :]))

        mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)
        variance = np.var(dropout_predictions, axis=0)  # shape (n_samples, n_classes)
        # pred_entropy = predictive_entropy(dropout_predictions)

        same_img = [cv2.imread(path)[:,:,::-1] for path in list_paths]
        for idx_img, (image, m, v) in enumerate(zip(same_img, mean, variance)):
            pred_idx = m.argmax()
            axes[0, idx_img].imshow(image)
            axes[0, idx_img].set_title(f"{list_classes[pred_idx]}\n{m[pred_idx]:.2f}-{v[pred_idx]:.4f}")
            axes[0, idx_img].axis('off')

        img_id = list_paths[0].split("/")[-1].split("_")[1]
        plt.suptitle(
            f"Filename: {list_paths[0].split('/')[-1]}\nGround Truth: {df.loc[df['img_id'] == int(img_id), 'annotation'].item()}",
            fontweight='bold')

        y_true.append(list_classes.index(df.loc[df['img_id'] == int(img_id), 'annotation'].item()))
        y_pred.append(m.argmax())
        conf.append(m.max())

        plt.tight_layout()
        plt.savefig(join(next_folder, f"{idx_annot}.png"))
        plt.close()

    plt.style.use("seaborn")

    plt.rc("font", size=12)
    plt.rc("axes", labelsize=12)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    plt.rc("legend", fontsize=12)
    plt.rc("axes", titlesize=16)
    plt.rc("figure", titlesize=16)

    fig = reliability_diagram(np.array(y_true), np.array(y_pred), np.array(conf), num_bins=10, draw_ece=True,
                        draw_bin_importance="alpha", draw_averages=True,
                        title="Reliability diagram", figsize=(6, 6), dpi=100,
                        return_fig=True)

    fig.savefig(join(next_folder, "reliability_diagram.png"))