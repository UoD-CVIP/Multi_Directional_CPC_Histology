# -*- coding: utf-8 -*-

"""

"""


# Built-in/Generic Imports
import os
import types

# Library Imports
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


__author__ = "Jacob Carse"
__copyright__ = "Copyright 2020, Multi-Directional Contrastive Predictive Coding"
__credits__ = ["Jacob Carse"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


class EfficientNetEncoder(nn.Module):
    """
    Class for the EfficientNet encoder model, containing the methods:
        init - Initialiser for the model.
        forward - Forward propagation for the encoder model.
    """

    def __init__(self, arguments):
        """
        Initialiser for the model that initialises hte models layers.
        :param arguments: Dictionary of arguments.
        """

        # Calls the super for the nn.Module.
        super(EfficientNetEncoder, self).__init__()

        # Gets the compound coefficient for EfficientNet.
        b = str(arguments['efficientnet_b'])

        # Loads the model with pretaining if selected.
        if arguments["pretraining"] == "imagenet":
            self.encoder = EfficientNet.from_pretrained(f"efficientnet-b{b}")
        else:
            self.encoder = EfficientNet.from_name(f"efficientnet-b{b}")

        # Pools the encoder outputs to a one dimensional array.
        self.encoder_pool = nn.AdaptiveAvgPool2d(1)

        # Gets the output dimensions of the encoder output.
        with torch.no_grad():
            temp_input = torch.zeros(1, 3, 96, 96)
            encoder_size = self.encoder.extract_features(temp_input).shape[1]

        # Initialises the code head for outputting feature vector.
        self.code_out = nn.Linear(encoder_size, arguments["code_size"])

    def forward(self, x):
        """
        Performs forward propagation with then EfficientNet encoder.
        :param x: PyTorch Tensor for the input image batch.
        :return: Feature vector output of the encoder.
        """

        # Performs feature extraction with the encoder model.
        x = self.encoder.extact_features(x)
        x = self.encoder_pool(x)
        x = x.view(x.shape[0], -1)
        return self.code_out(x)

    def save_model(self, path, name, epoch=None):
        """
        Method for saving the encoder model.
        :param path: Directory path to save the encoder.
        :param name: The name of the experiment to be saved.
        :param epoch: Integer for the current epoch to be included in the save name.
        """

        # Checks if the save directory exists and if not creates it.
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # Saves the model to the save directory.
        if epoch is not None:
            torch.save(self.state_dict(), os.path.join(path, f"{name}_encoder_{str(epoch)}.pt"))
        else:
            torch.save(self.state_dict(), os.path.join(path, f"{name}_encoder.pt"))
