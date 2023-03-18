# -*- coding: utf-8 -*-
"""
This script is in charge of training the model and evaluating it on the test set with the pytorch library.
https://pytorch.org/

The implemented architectures are:
    - VGG16, from the paper:
        "Very Deep Convolutional Networks for Large-Scale Image Recognition"
        https://arxiv.org/abs/1409.1556
    - ResNets, from the paper:
        "Deep Residual Learning for Image Recognition"
        https://arxiv.org/abs/1512.03385
    - EfficientNets, from the paper:
        "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
        https://arxiv.org/abs/1905.11946
    - EfficientNetsV2, from the paper:
        "EfficientNetV2: Smaller Models and Faster Training"
        https://arxiv.org/abs/2104.00298
    - ConvNeXts, from the paper:
        "A ConvNet for the 2020s"
        https://arxiv.org/abs/2201.03545

The training uses:
    - Adam optimizer:
        "Adam: A Method for Stochastic Optimization
        https://arxiv.org/abs/1412.6980"
        https://pytorch.org/docs/stable/optim.html
    - CrossEntropyLoss
    - Grad Scaler:

@author: David Serrano Lozano, @davidserra9
"""

import timm
import time
import wandb
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.NN_utils import *
from src.ICM_dataset import ICMDataset
from utils.config_parser import load_yml
from utils.inference_utils import inference_fn

def main():

    # Find which device is used
    if torch.cuda.is_available():
        print(f'Training the model in {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        print('CAREFUL!! Training the model with CPU')

    # Load the model
    model = timm.create_model("resnet50", pretrained=True, num_classes=6)
    model = model.to("cuda")

    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Initialize the model
    loss_fn = nn.CrossEntropyLoss()  # Initialize the loss function
    scaler = torch.cuda.amp.GradScaler()  # Initialize the scaler

    train_dataset = ICMDataset(path="/media/david/media/TFM/article_dataset/train/",
                               train=True,
                               oversample=True,
                               species=["spatangus_purpureus",
                                        "echinaster_sepositus",
                                        "cerianthus_membranaceus",
                                        "bonellia_viridis",
                                        "scyliorhinus_canicula",
                                        "ophiura_ophiura",
                                        "background"])

    test_dataset = ICMDataset(path="/media/david/media/TFM/article_dataset/test/",
                              train=False,
                              oversample=False,
                              species=["spatangus_purpureus",
                                       "echinaster_sepositus",
                                       "cerianthus_membranaceus",
                                       "bonellia_viridis",
                                       "scyliorhinus_canicula",
                                       "ophiura_ophiura",
                                       "background"])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Initialize the metrics dictionaries
    train_metrics = {'accuracy': [], 'loss': []}
    test_metrics = {'accuracy': [], 'loss': [], 'f1': []}

    # Training loop
    for epoch in range(10):

        train_acc, train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, "cuda", epoch)  # Train
        train_metrics['accuracy'].append(train_acc)  # Append train accuracy
        train_metrics['loss'].append(train_loss)  # Append train accuracy

        test_acc, test_loss, test_f1 = eval_fn(test_loader, model, loss_fn, "cuda", epoch)  # Validate the model
        test_metrics['accuracy'].append(test_acc)  # Append test accuracy
        test_metrics['loss'].append(test_loss)  # Append test loss
        test_metrics['f1'].append(test_f1)  # Append test f1 score

        print(f'Epoch {epoch + 1}/{10} - Train accuracy: {train_acc:.2f} - Train loss: {train_loss:.2f} - Test accuracy: {test_acc:.2f} - Test loss: {test_loss:.2f} - Test f1: {test_f1:.2f}')


if __name__ == '__main__':
    main()
