# -*- coding: utf-8 -*-
"""
This module contains the functions to train, evaluate, load and save the model.
@author: David Serrano Lozano, @davidserra9
"""

from os.path import join
import albumentations as A
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from beautifultable import BeautifulTable
from packaging import version
from sklearn.metrics import f1_score
from tqdm import tqdm


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
            folder/root path where the model will be saved

        Returns
        -------
        model : object
    """

    # Implemented architectures
    IMP_ARCH = ['vgg16',
                'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
                'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l',
                'convnext_tiny', 'convnext_large']
    try:
        if model_name in IMP_ARCH:
            if 'vgg' in model_name:
                # Handle possible changes in torchvision versions
                if version.parse(torchvision.__version__) >= version.parse("0.13.0"):
                    model = getattr(models, model_name)(weights="IMAGENET1K_V1")

                else:
                    model = getattr(models, model_name)(pretrained=True)

                num_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_features, num_classes)
                model.name = model_name

            elif 'resnet' in model_name:
                # Handle possible changes in torchvision versions
                if version.parse(torchvision.__version__) >= version.parse("0.13.0"):
                    model = getattr(models, model_name)(weights="IMAGENET1K_V1")

                else:
                    model = getattr(models, model_name)(pretrained=True)

                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
                model.name = model_name

            elif 'efficientnet_b' in model_name:
                # Handle possible changes in torchvision versions
                if version.parse(torchvision.__version__) >= version.parse("0.13.0"):
                    model = getattr(models, model_name)(weights="IMAGENET1K_V1")
                elif version.parse(torchvision.__version__) >= version.parse("0.11.0"):
                    model = getattr(models, model_name)(pretrained=True)
                else:
                    raise ValueError('EfficientNet requires torchvision >= 0.11.0! :(')

                num_ftrs = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
                model.name = model_name

            elif 'efficientnet_v2_' in model_name:
                # Handle possible changes in torchvision versions
                if version.parse(torchvision.__version__) >= version.parse("0.13.0"):
                    model = getattr(models, model_name)(weights="IMAGENET1K_V1")
                else:
                    raise ValueError('ConvNext requires torchvision >= 0.13.0! :(')

                num_ftrs = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
                model.name = model_name

            elif 'convnext' in model_name:
                # Handle possible changes in torchvision versions
                if version.parse(torchvision.__version__) >= version.parse("0.13.0"):
                    model = getattr(models, model_name)(weights="IMAGENET1K_V1")
                elif version.parse(torchvision.__version__) >= version.parse("0.12.0"):
                    model = getattr(models, model_name)(pretrained=True)
                else:
                    raise ValueError('ConvNext requires torchvision >= 0.12.0! :(')

                num_ftrs = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
                model.name = model_name

        else:
            raise ValueError('Model not implemented')

    except Exception as e:
        print(e)
        print(f'Model {model_name} not found. Please check the model name or torchvision version (0.13.0).')
        print("Models implemented:"
              "\nVGG16"
              "\nResNets: resnet18, resnet50..."
              "\nEfficientNets: efficientnet_b0, efficientnet_b1..."""
              "\nEfficientNetsV2: efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l"
              "\nConvNeXts: convnext_tiny, convnext_large"
              )
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
        bt.rows.append([model_name, checkpoint['epoch'], f"{checkpoint['test_acc']:.4f}", f"{checkpoint['f1']:.4f}"])
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


def get_training_augmentations():
    """ Function defining and returning the training augmentations.

    Returns
    -------
    train_transform : albumentations.Compose
        training augmentations
    """
    train_transform = [
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
    ]
    return A.Compose(train_transform)


def get_validation_augmentations():
    """ Function defining and returning the validation/test augmentations.

        Returns
        -------
        val_transforms : albumentations.Compose
            training augmentations
    """
    val_transforms = [
        A.Normalize(mean=[0.4493, 0.5078, 0.4237],
                    std=[0.1263, 0.1265, 0.1169]),
        A.Resize(224, 224),
        ToTensorV2(),
    ]
    return A.Compose(val_transforms)


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

    loop = tqdm(loader,  # Create the tqdm bar for visualizing the progress
                desc=f'EPOCH {epoch_num} TRAIN',
                leave=True)

    correct = 0  # accumulated correct predictions
    total_samples = 0  # accumulated total predictions
    loss_sum = 0  # accumulated loss

    for idx, (data, targets) in enumerate(loop):
        data, targets = data.to(device), targets.to(device)  # data and labels to device
        optimizer.zero_grad()  # Initialize gradients

        outputs = model(data)  # Forward pass
        loss = loss_fn(outputs, targets)  # Compute the loss
        _, predictions = torch.max(outputs.data, 1)  # Obtain the classes with higher probability

        total_samples += data.size(0)  # Subtotal of the predictions
        correct += (predictions == targets).sum().item()  # Subtotal of the correct predictions
        loss_sum += loss.item()  # Subtotal of the correct losses

        scaler.scale(loss).backward()  # Backward pass
        scaler.step(optimizer)  # Update the weights
        scaler.update()  # Update the scale

        loop.set_postfix(acc=correct / total_samples,
                         loss=loss_sum / (idx + 1))

    epoch_acc = correct / total_samples  # Epoch accuracy
    epoch_loss = loss_sum / len(loader)  # Epoch loss

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

    loop = tqdm(loader,  # Create the tqdm bar for visualizing the progress
                desc=f'EPOCH {epoch_num}  TEST',
                leave=True)

    correct = 0  # Accumulated correct predictions
    total_samples = 0  # Accumulated total predictions
    loss_sum = 0  # Accumulated loss
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

            loss = loss_fn(outputs, targets)  # Compute loss

            _, predictions = torch.max(outputs.data, 1)  # Obtain class with max probability

            y_true = y_true + targets.tolist()  # Add the batch ground truth labels
            y_pred = y_pred + predictions.tolist()  # Add the batch predicted labels

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
