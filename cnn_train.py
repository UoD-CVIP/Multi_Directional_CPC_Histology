## -*- coding: utf-8 -*-

"""
This file contains implementations of the functions used to train a CNN model:
    train_cnn - Function used to facilitate the training of the Convolutinal Neural Network model.
    test_cnn - Function used for the testing of training of the Convolutinal Neural Network model.
"""


# Built-in/Generic Imports
import os
import time

# Library Imports
from torch.utils.tensorboard import SummaryWriter

# Own Modules Imports
from utils import *
from models import *
from dataset import *


__author__ = "Jacob Carse"
__copyright__ = "Copyright 2020, Multi-Directional Contrastive Predictive Coding for Histology"
__credits__ = ["Jacob Carse", "Stephen McKenna"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


def train_cnn(arguments, device):
    """
    Function used to train the Contrastive Predictive Coding Model.
    :param arguments: Dictionary containing arguments.
    :param device: PyTorch device object.
    :return: Returns lists of training and validation losses and an integer for the best performing epoch.
    """

    # Loads a TensorBoard Summary Writer.
    if arguments["tensorboard"]:
        writer = SummaryWriter(os.path.join("TensorBoard", arguments["task"], arguments["experiment"]))

    # Loads the training data and reduces its size.
    train_data = Dataset(arguments, "train")
    train_data.reduce_size(arguments["training_examples"])

    # Splits the training data into a validation set.
    validation_data = train_data.get_validation_set()

    # Creates the data loaders for the training and validation data.
    training_data_loader = DataLoader(train_data, batch_size=arguments["batch_size"],
                                           shuffle=True, num_workers=arguments["data_workers"],
                                           pin_memory=False, drop_last=True)
    validation_data_loader = DataLoader(validation_data, batch_size=arguments["batch_size"],
                                          shuffle=False, num_workers=arguments["data_workers"],
                                          pin_memory=False, drop_last=True)
