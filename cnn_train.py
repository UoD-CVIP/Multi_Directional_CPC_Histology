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
import torch
from apex import amp
from torch import optim
from torch.utils.data import DataLoader
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

    log(arguments, "Loaded Datasets")

    # Initialises the encoder and autoregressor.
    encoder = Encoder(0, arguments["image_size"], imagenet=arguments["pretrained"].lower() == "imagenet")
    classifier = Classifier(encoder.encoder_size, 10, arguments["hidden_layer"])

    # Loads weights from pretrained Contrastive Predictive Coding model.
    if arguments["pretrained"].lower() == "cpc":
        encoder_path = os.path.join(arguments["model_dir"], f"{arguments['experiment']}_encoder_best.pt")
        encoder.load_state_dict(torch.load(encoder_path, map_location=device), strict=False)

    # Sets the models to training mode.
    encoder.train()
    classifier.train()

    # Moves the models to the selected device.
    encoder.to(device)
    classifier.to(device)

    # Combines the parameters from the two models.
    parameters = list(encoder.parameters()) + list(classifier.parameters())

    # Initialises a optimiser used to optimise the parameters of the models.
    optimiser = optim.Adam(params=parameters, lr=arguments["learning_rate"])

    # If 16 bit precision is being used change the model and optimiser precision.
    if arguments["precision"] == 16:
        [encoder, classifier], optimiser = amp.initialize([encoder, classifier], optimiser,
                                                          opt_level="O2", verbosity=False)

    # Checks if precision level is supported and if not defaults to 32.
    elif arguments["precision"] != 32:
        log(arguments, "Only 16 and 32 bit precision supported. Defaulting to 32 bit precision.")

    log(arguments, "Models Initialised")
