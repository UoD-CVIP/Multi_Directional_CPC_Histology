## -*- coding: utf-8 -*-

"""

"""


# Built-in/Generic Imports
import os

# Library Imports
from torch.utils.data import DataLoader

# Own Modules Imports
from utils import *
from models import Encoder
from dataset import Dataset


__author__ = "Jacob Carse"
__copyright__ = "Copyright 2020, Multi-Directional Contrastive Predictive Coding for Histology"
__credits__ = ["Jacob Carse", "Stephen McKenna"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


def make_representations(arguments, device):
    """
    Creates representations for all data.
    :param arguments: Dictionary containing arguments.
    :param device: PyTorch device object.
    """

    # Loads training and testing data.
    train_data = Dataset(arguments, "train")
    test_data = Dataset(arguments, "test")

    # Creates the data loaders for the training and testing data.
    training_data_loader = DataLoader(train_data, batch_size=arguments["batch_size"],
                                      shuffle=False, num_workers=arguments["data_workers"],
                                      pin_memory=False, drop_last=False)
    testing_data_loader = DataLoader(test_data, batch_size=arguments["batch_size"],
                                      shuffle=False, num_workers=arguments["data_workers"],
                                      pin_memory=False, drop_last=False)

    log(arguments, "Loaded Datasets")

    # Initialises the encoder.
    encoder = Encoder(0, arguments["image_size"], arguments["pretrained"] == "imagenet")

    # Loads weights from pretrained Contrastive Predictive Coding model.
    if arguments["pretrained"].lower() == "cpc":
        encoder_path = os.path.join(arguments["model_dir"], f"{arguments['experiment']}_encoder_best.pt")
        encoder.load_state_dict(torch.load(encoder_path, map_location=device), strict=False)

    # Sets the model to evaluation mode.
    encoder.eval()

    # Moves the model to the selected device.
    encoder.to(device)

    # If 16 bit precision is being used change the model and optimiser precision.
    if arguments["precision"] == 16:
        encoder = amp.initialize(encoder, opt_level="O2", verbosity=False)

    # Checks if precision level is supported and if not defaults to 32.
    elif arguments["precision"] != 32:
        log(arguments, "Only 16 and 32 bit precision supported. Defaulting to 32 bit precision.")

    log(arguments, "Models Initialised")

    # Creates the HDF5 files used to store the training and testing data representations.
    train_representations = HDF5Handler(os.path.join(arguments["representation_dir"],
                                                     f"{arguments['experiment']}_train.h5"),
                                        'x', encoder.encoder_size)
    test_representations = HDF5Handler(os.path.join(arguments["representation_dir"],
                                                    f"{arguments['experiment']}_test.h5"),
                                        'x', encoder.encoder_size)

    train_labels, test_labels = [], []

    log(arguments, "HDF5 Representation Files Created.")
