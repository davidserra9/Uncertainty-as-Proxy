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

import time
import wandb
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.NN_utils import *
from utils.UW_dataset import UWDataset
from utils.config_parser import load_yml
from utils.inference_utils import inference_fn

def main():
    """ Main function of the model (training and evaluation) """

    cfg = load_yml("config.yml")

    # Find which device is used
    if torch.cuda.is_available() and cfg.device == "cuda":
        print(f'Training the model in {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        print('CAREFUL!! Training the model with CPU')

    wandb.init(project="new-exp",
               entity="tfm",
               name=cfg.species_classification.model,
               config=dict(learning_rate=cfg.species_classification.learning_rate,
                           architecture=cfg.species_classification.model,
                           epochs=cfg.species_classification.num_epochs,
                           batch_size=cfg.species_classification.batch_size,
                           ))

    # Initialize the model
    model = initialize_model(model_name=cfg.species_classification.model,
                             num_classes=len(cfg.species),
                             load_model=cfg.species_classification.load_model,
                             model_root=cfg.model_path)
    model.to(cfg.device)

    # Initialize optimizer, loss and scaler
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.species_classification.learning_rate)  # Initialize the model
    loss_fn = nn.CrossEntropyLoss()  # Initialize the loss
    scaler = torch.cuda.amp.GradScaler()  # Initialize the Scaler

    # Initialize datasets
    train_dataset = UWDataset(split_list=[join(cfg.species_dataset, "train_images")],
                              # join(cfg.species_dataset, "val_images")],
                              list_classes=cfg.species,
                              train=True,
                              img_per_annot=cfg.species_classification.img_per_annot)

    test_dataset = UWDataset(split_list=[join(cfg.species_dataset, "test_images")],
                             list_classes=cfg.species,
                             train=False)

    # Initialize dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.species_classification.batch_size,
                              num_workers=cfg.species_classification.num_workers,
                              pin_memory=True,
                              shuffle=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.species_classification.batch_size,
                             num_workers=cfg.species_classification.num_workers,
                             pin_memory=True)

    # Initialize the metrics dictionaries
    train_metrics = {'accuracy': [], 'loss': []}
    test_metrics = {'accuracy': [], 'loss': [], 'f1': []}

    print("")
    print("----------- MODEL: {} --------------".format(model.name))
    print("----------- TRAINING START --------------")
    print("")
    time.sleep(1)

    # Training loop
    for epoch in range(cfg.species_classification.num_epochs):

        train_acc, train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, cfg.device, epoch)  # Train
        train_metrics['accuracy'].append(train_acc)  # Append train accuracy
        train_metrics['loss'].append(train_loss)  # Append train accuracy

        test_acc, test_loss, test_f1 = eval_fn(test_loader, model, loss_fn, cfg.device, epoch)  # Validate the model
        test_metrics['accuracy'].append(test_acc)  # Append test accuracy
        test_metrics['loss'].append(test_loss)  # Append test loss
        test_metrics['f1'].append(test_f1)  # Append test f1 score

        # If the validation accuracy is the best one so far, save the model
        if (test_f1 == max(test_metrics['f1'])) and cfg.species_classification.save_model:
            save_model(model=model,
                       optimizer=optimizer,
                       num_epoch=epoch,
                       acc=test_acc,
                       f1=test_f1,
                       model_root=cfg.model_path)

        # Refresh wandb
        wandb.log({"train_loss": train_loss,
                   "train_accuracy": train_acc,
                   "test_loss": test_loss,
                   "test_accuracy": test_acc,
                   "test_f1": test_f1})

    # Once the training has ended, run inference on the best weights
    print("")
    print("----------- MODEL: {} --------------".format(model.__class__.__name__))
    print("----------- INFERENCE START --------------")
    print("")

    model = initialize_model(model_name=cfg.species_classification.model,
                             num_classes=len(cfg.species),
                             load_model=True,
                             model_root=cfg.model_path)
    model.to(cfg.device)

    inference_fn(model=model,
                 loader=test_loader,
                 output_root=cfg.output_path,
                 list_classes=cfg.species,
                 mc_samples=50,
                 device=cfg.device,
                 cm=True,
                 uncertainty=True)


if __name__ == '__main__':
    main()
