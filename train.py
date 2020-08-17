# -*- coding: utf-8 -*-

"""
This file contains implementations of the functions used to train the multi directional CPC model:
    train_cpc - Function used to facilitate the training of the Multi Directional Contrastive Predictive Coding model.
"""


# Built-in/Generic Imports

# Library Imports
from torch.utils.data import DataLoader

# Own Modules Imports
from dataset import *


__author__ = "Jacob Carse"
__copyright__ = "Copyright 2020, Multi-Directional Contrastive Predictive Coding for Histology"
__credits__ = ["Jacob Carse", "Stephen McKenna"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


def train_cpc(arguments, device):
    """
    Function used to train the Multi-Directional Contrastive Predictive Coding model.
    :param arguments: Dictonary containing arguments.
    :param device: PyTorch device object.
    """

    # Loads the training data and splits it into a validation data.
    train_data = Dataset(arguments, "train")
    validation_data = train_data.get_validation_set()

    # Creates the data loaders for the training and validation data.
    training_data_loader = DataLoader(train_data, batch_size=arguments["batch_size"],
                                           shuffle=True, num_workers=arguments["data_workers"],
                                           pin_memory=False, drop_last=True)
    validation_data_loader = DataLoader(validation_data, batch_size=arguments["batch_size"],
                                          shuffle=False, num_workers=arguments["data_workers"],
                                          pin_memory=False, drop_last=True)

    # Shuffles the training and validation data.
    random_train_data = train_data.shuffle()
    random_validation_data = validation_data.shuffle()

    # Creates the random data loaders for the training and validation data.
    random_training_loader = DataLoader(random_train_data, batch_size=arguments["random_patches"],
                                             shuffle=True, num_workers=arguments["data_workers"],
                                             pin_memory=False, drop_last=True)

    random_validation_loader = DataLoader(random_validation_data, batch_size=arguments["random_patches"],
                                               shuffle=True, num_workers=arguments["data_workers"],
                                               pin_memory=False, drop_last=True)
