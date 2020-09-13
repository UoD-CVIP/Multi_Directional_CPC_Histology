#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
This file is the main executable for Multi-Directional Contrastive Predictive Coding for Histology.
This file loads the arguments, sets random seed and runs a selected task.
"""


# Own Module Import
from cpc_train import *
from utils import *
from config import *


__author__ = "Jacob Carse"
__copyright__ = "Copyright 2020, Multi-Directional Contrastive Predictive Coding for Histology"
__credits__ = ["Jacob Carse", "Stephen McKenna"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


if __name__ == "__main__":
    # Loads the arguments from a config file and command line arguments.
    description = "An implementation of Multi-Directional Contrastive Predictive Coding for Histology."
    arguments = load_arguments(description)
    log(arguments, "Loaded Arguments:")
    print_arguments(arguments)

    # Sets the random seed if specified.
    if arguments["seed"] != 0:
        set_random_seed(arguments["seed"])
        log(arguments, f"Set Random Seed to {arguments['seed']}")

    # Gets device that is used for training the model.
    device = get_device(arguments)
    log(arguments, f"Running on Device: {device}")

    # Trains and Tests a Contrastive Predictive Coding model.
    if arguments["task"].lower() == "cpc":
        train_cpc(arguments, device)
        test_cpc(arguments, device)

    # Trains a Contrastive Predictive Coding Model.
    elif arguments["task"].lower() == "train_cpc":
        train_cpc(arguments, device)

    # Tests a Contrastive Predictive Coding Model.
    elif arguments["task"].lower() == "test_cpc":
        test_cpc(arguments, device)
