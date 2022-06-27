import os
import albumentations as A
import cv2
import numpy as np
import seaborn as sns
import random
import torch
import torch.nn as nn
from os.path import join
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import confusion_matrix
from albumentations.pytorch import ToTensorV2

def initialize_model(model_name, num_classes, load_model, balance, data_aug, model_root):
    """ Function to initialize the model depending on the desired architecture.

        Parameters
        ----------
        model_name : string
            name of the model to initialize
        num_classes : int
            number of classes to predict
        load_model : boolean
            if True, load the model from the disk
        balance : string
            when loading an existing model, undersampling, oversampling or other
        data_aug : boolean
            when loading an existing model, data augmentation when training or not

        Returns
        -------
        model : object
    """

    # Initialize the desired architecture and change the last linear layer to fit the number of classes
    if 'resnet' in model_name:
        model = torch.hub.load('pytorch/vision:v0.8.0', model_name, pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.name = model_name

    else:
        raise Exception('Wrong model name: {}'.format(model_name))

    # If load_model==True, load the weights of the model
    if load_model:
        if balance in ["oversampling", "undersampling"]:
            if data_aug:
                model_path = join(model_root, "_".join([model_name, balance, "DA"]) + ".pth.tar")
            else:
                model_path = join(model_root, "_".join([model_name, balance]) + ".pth.tar")

        else:
            if data_aug:
                model_path = join(model_root, "_".join([model_name, "DA"]) + ".pth.tar")
            else:
                model_path = join(model_root, model_name + ".pth.tar")

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded from {}'.format(model_path))
    else:
        print('Model initialized')

    return model

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

def pred_fn(folder_path, model, list_classes, n_images, output_path, device) -> None:
    """ Function to run inference on some images of the test set

        Parameters
        ----------
        folder_path : str
            path to the folder containing the test images
        model : object
            pytorch model to run inference
        list_classes : list
            list of the names of the classes
        n_images : int
            number of images to run inference on (taken randomly)
        output_path : str
            path to the folder where the output files will be saved
            (.../experiments/infxx/inference/)
        device : str
            'cuda' or 'cpu'
    """

    model.eval()                                    # ensure that the model is in eval mode

    all_images = glob(join(folder_path, "*.jpg"))   # Get all the images in the folder
    random.shuffle(all_images)                      # Shuffle the image paths
    query_images = all_images[:n_images]            # Get the first 20 images (randomly chosen)

    transformations = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.4493, 0.5078, 0.4237],
                            std=[0.1263, 0.1265, 0.1169]),
                ToTensorV2()
            ])

    for idx_query, im_path in tqdm(enumerate(query_images), desc="Running inference"):
        same_annot = [cv2.imread(path)[:, :, ::-1] for path in sorted(all_images) if im_path[:-5] in path]

        num_img = len(same_annot)
        image_outputs = []
        plt.figure(figsize=(20, 5))
        for idx, image in enumerate(same_annot):
            plt.subplot(1, num_img, idx+1)
            plt.imshow(image)
            plt.axis('off')

            outputs = model(transformations(image=image)["image"].unsqueeze(0).to(device))
            image_outputs.append(outputs)

            _, predictions = torch.max(outputs.data, 1)
            plt.title(f'Frame pred: {list_classes[predictions.item()]}')

        mean_outputs = torch.mean(torch.stack(image_outputs), dim=0)
        _, predictions = torch.max(mean_outputs.data, 1)
        plt.suptitle(f"Ground Truth: {im_path.split('/')[-1].split('_')[0]}\nPred: {list_classes[predictions.item()]}",
                     fontweight ="bold")
        plt.tight_layout()
        plt.savefig(join(output_path, f"{output_path.split('/')[-2]}_{idx_query:02}.png"))
        plt.close()

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

    cm = plot_cm(y_true, y_pred, list_classes)
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
        if isinstance(module, nn.ReLU):
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

def get_monte_carlo_predictions(folder_path, forward_passes, model, list_classes, n_images, output_path, device) -> None:
    """ Function to get the MonteCarlo samples and Uncertainty Estimates
        through multiple forward passes
        (https://stackoverflow.com/a/63397197/15098668)

        Parameters
        ----------
        folder_path : str
            string of the folder path which contains the inference images
        forward_passes : int
            number of monte-carlo samples/forward passes
        model : object
            pytorch model
        list_classes : list
            list with the name of the classes
        n_images : int
            number of samples to run the example (taken randomly)
        output_path : str
            string of the folder path where the output files will be saved
        device : str
            'cuda' or 'cpu'
    """

    model.eval()                        # Ensure that the model is in evaluation mode
    append_dropout(model)               # Append the dropout layers
    enable_dropout(model)               # Enable the dropout layers (train mode)
    model.to(device)

    dropout_predictions = np.empty((0, n_images, len(list_classes)))
    softmax = nn.Softmax(dim=1)

    all_images = glob(join(folder_path, "*.jpg"))   # Get all the images in the folder
    random.shuffle(all_images)                      # Shuffle the image paths
    query_images = all_images[:n_images]           # Get the first 20 images (randomly chosen)

    # Find all the images of the same annotation
    same_annot = [[r for r in sorted(all_images) if q[:-5] in r] for q in query_images]

    transformations = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.4493, 0.5078, 0.4237],
                            std=[0.1263, 0.1265, 0.1169]),
                ToTensorV2()
            ])

    for _ in tqdm(range(forward_passes), desc="MC Dropout"):
        predictions = np.empty((0, len(list_classes)))
        enable_dropout(model)

        for same_list in same_annot:
            # Get all the images of the same annotation
            images = [transformations(image=cv2.imread(img_path)[:,:,::-1])["image"] for img_path in same_list]
            images = torch.stack(images).to(device)
            with torch.no_grad():
                output = torch.mean(softmax(model(images)), dim=0)

            predictions = np.vstack((predictions, output.cpu().numpy()))

        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))

    mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)
    variance = np.var(dropout_predictions, axis=0) # shape (n_samples, n_classes)

    for idx_query, same_list in enumerate(same_annot):
        same_img = [cv2.imread(path)[:,:,::-1] for path in same_list]
        num_imgs = len(same_list)

        plt.figure(figsize=(20, 5))
        for idx, image in enumerate(same_img):
            plt.subplot(1, num_imgs, idx+1)
            plt.imshow(image)
            plt.axis('off')

        pred_label = list_classes[np.argmax(mean, axis=1)[idx_query]]
        pred_prob = np.max(mean, axis=1)[idx_query]

        plt.suptitle(f"Ground Truth: {same_list[0].split('/')[-1].split('_')[0]}\nPred: {pred_label} - {pred_prob:.2f}",
                      fontweight='bold')
        plt.tight_layout()
        plt.savefig(join(output_path, f"{output_path.split('/')[-2]}_{idx_query:02}.png"))
        plt.close()

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

    random.seed(42)

    # Create the folder in which the output files will be saved
    os.makedirs(output_root, exist_ok=True)
    next_folder = len(os.listdir(output_root)) + 1
    next_folder = join(output_root, f"inf{next_folder:02}")
    os.makedirs(next_folder)
    os.makedirs(join(next_folder, "inference"))
    os.makedirs(join(next_folder, "MC_dropout"))

    model.eval()                        # Ensure that the model is in evaluation mode

    pred_fn(folder_path=folder_path,
            model=model,
            list_classes=list_classes,
            n_images=n_images,
            output_path=join(next_folder, "inference"),
            device=device)

    confusion_matrix_fn(loader=loader,
                        model=model,
                        list_classes=list_classes,
                        output_path=next_folder,
                        device=device)

    get_monte_carlo_predictions(folder_path=folder_path,
                                forward_passes=n_mc_samples,
                                model=model,
                                list_classes=list_classes,
                                n_images=50,
                                output_path=join(next_folder, "MC_dropout"),
                                device=device)

def save_model(model, optimizer, num_epoch, acc, f1, model_root, balance, data_aug):
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
        balance : str
            'oversampling', 'undersampling' or other to identify how the model has been trained
        data_aug : bool
            True if the data augmentation has been used to train the model
    """

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": num_epoch,
        "test_acc": acc,
        "f1": f1
    }

    # Model path
    if balance in ["oversampling", "undersampling"]:
        if data_aug:
            model_path = join(model_root, "_".join([model.name, balance, "DA"]) + ".pth.tar")
        else:
            model_path = join(model_root, "_".join([model.name, balance]) + ".pth.tar")

    else:
        if data_aug:
            model_path = join(model_root, "_".join([model.name, "DA"]) + ".pth.tar")
        else:
            model_path = join(model_root, model.name + ".pth.tar")

    torch.save(checkpoint, model_path)
    print("Model saved...")
