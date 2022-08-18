import torch
import numpy as np
import torch.nn as nn
from os.path import join
from torch.utils.data import DataLoader

from utils.UW_dataset import UWDataset
from utils.config_parser import load_yml
from utils.NN_utils import initialize_model

class MCDP_model(object):
    """Monte-Carlo Dropout model wrapper
    Implemented Architectures: (if the architecture does not have a dropout layer, then it is added at inference time)
        - VGG
        - ResNet
        - EfficientNet
        - EfficientNetV2
        - ConvNeXt
    """

    def __init__(self, model, num_classes, mc_samples=25, dropout_rate=0.5, device="cuda"):

        self.model = model.eval()
        self.dropout_rate = dropout_rate
        self.samples = mc_samples
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=2)
        self.device = device

        # VGG: dropout layer in train mode
        if 'vgg' in model.name:
            self.model = self.train_dropout(self.model)

        # ResNet: append dropout layer in the classifier in train mode
        elif 'resnet' in model.name:
            self.model = self.append_dropout_resnet(self.model)
            self.model = self.train_dropout(self.model)

        # EfficientNet: dropout layer in train mode
        elif 'efficientnet_b' in model.name:
            self.model = self.train_dropout(self.model)

        # EfficientNetV2: dropout layer in train mode
        elif 'effificnet_v2_b' in model.name:
            self.model = self.train_dropout(self.model)

        # ConvNeXt: Add dropout layer in the classifier in train mode
        elif 'convnext' in model.name:
            self.model.classifier = self.change_dropout_rate(self.model.classifier)
            self.model = self.train_dropout(self.model)

    def append_dropout_resnet(self, model):
        """ Append dropout layer to the resnet classifier

            Parameters
            ----------
            model : torch.nn.Module
                ResNet model (with classifier module)
        """

        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                self.append_dropout_resnet(module)
            if name == 'layer4':
                new = nn.Sequential(module, nn.Dropout2d(p=self.dropout_rate, inplace=True))
                setattr(model, name, new)
        return model

    def train_dropout(self, model):
        """ Function to put dropout layers in training mode

            Parameters
            ----------
            model : torch.nn.Module
        """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        return model

    def change_dropout_rate(self, model):
        """ Change the dropout rate of the model

            Parameters
            ----------
            model : torch.nn.Module
        """
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                self.change_dropout_rate(module)
            if name == 'drop':
                new = nn.Sequential(module, nn.Dropout2d(p=self.dropout_rate, inplace=True))
                setattr(model, name, new)
        return model

    def forward(self, x):
        """ N forward passes of the model

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, N, C, H, W)
            B: batch size
            N: number of images per annotation (could be 1 if 1 img/annot)
            C: number of channels
            H: height
            W: width

        Return
        ------
        np.array
            Array with shape (B, S, N, L)
            B: batch size or number of images in the dataloader
            S: number of monte-carlo samples
            N: number of images per annotation (could be 1 if 1 img/annot)
            L: number of classes
            (If 1 image per annotation, then S = 1)
        """

        dropout_predictions = np.empty((0, x.shape[0], x.shape[1], self.num_classes))
        for _ in range(self.samples):
            with torch.no_grad():
                if len(x.shape) == 4:
                    x = x.unsequeeze(1)

                outputs = torch.stack([self.model(x[i, :, :, :, :].to(self.device)) for i in range(x.shape[0])])
                outputs = self.softmax(outputs)

            dropout_predictions = np.vstack((dropout_predictions, outputs.cpu().numpy()[np.newaxis, :, :]))

        return dropout_predictions.transpose(1, 0, 2, 3)

    def __call__(self, x):
        return self.forward(x)

if __name__ == "__main__":
    cfg = load_yml("../config.yml")

    # Initialize the model
    model = initialize_model(model_name=cfg.species_classification.model,
                             num_classes=len(cfg.species),
                             load_model=True,
                             model_root=cfg.model_path)

    model.to(cfg.device)

    # Initialize datasets
    test_dataset = UWDataset(split_list=[join(cfg.species_dataset, "test_images")],
                             list_classes=cfg.species,
                             train=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.species_classification.batch_size,
                             num_workers=cfg.species_classification.num_workers,
                             pin_memory=True,
                             shuffle=True)

    print(len(test_loader))

    MCDP_wrapper = MCDP_model(model, len(cfg.species), mc_samples=25, dropout_rate=0.5)

    batch = next(iter(test_loader))[0]
    for (data, labels) in test_loader:
        data = data.to(cfg.device)
        pred = MCDP_wrapper(data)
        print()
