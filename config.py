# -*- coding: utf-8 -*-

"""
This file contains the function to read arguments from a config file and command line.
    load_arguments - Function to load arguments from the config file and command line.
"""


# Built-in/Generic Imports
import sys
from argparse import ArgumentParser
from configparser import ConfigParser

# Own Module Import
from utils import *


__author__ = "Jacob Carse"
__copyright__ = "Copyright 2020, Multi-Directional Contrastive Predictive Coding for Histology"
__credits__ = ["Jacob Carse", "Stephen McKenna"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


def load_arguments(description):
    """
    Loads arguments from a config file and command line.
    Arguments from command line overrides arguments from the config file.
    The config file will be loaded from the default location ./config.ini and can be overridden from the command line.
    :param description: The description of the application.
    :return: Dictionary of arguments.
    """

    # Creates a ArgumentParser to read command line arguments.
    argument_parser = ArgumentParser(description=description)

    # Creates a ConfigParser to read the config file.
    config_parser = ConfigParser()

    # Loads either a specified config file or default config file.
    if len(sys.argv) > 1:
        if sys.argv[1] == "--config_file":
            config_parser.read(sys.argv[2])
        else:
            config_parser.read("config.ini")
    else:
        config_parser.read("config.ini")

    # Standard Arguments
    argument_parser.add_argument("--config_file", type=str,
                                 default="config.ini",
                                 help="String representing the file path to the config file.")
    argument_parser.add_argument("--task", type=str,
                                 default=config_parser["standard"]["task"],
                                 help="String representing the task for the application to run.")
    argument_parser.add_argument("--seed", type=int,
                                 default=int(config_parser["standard"]["seed"]),
                                 help="Integer for the random seed.")
    argument_parser.add_argument("--experiment", type=str,
                                 default=config_parser["standard"]["experiment"],
                                 help="String representing the name of the current experiment.")

    # Logging Arguments
    argument_parser.add_argument("--log_dir", type=str,
                                 default=config_parser["logging"]["log_dir"],
                                 help="Directory where the log files will be stored.")
    argument_parser.add_argument("--tensorboard", type=str_to_bool,
                                 default=config_parser["logging"]["tensorboard"].lower() == "true",
                                 help="Boolean value for if TensorBoard logging should be used.")
    argument_parser.add_argument("--log_intervals", type=int,
                                 default=int(config_parser["logging"]["log_intervals"]),
                                 help="Integer for the number of batches before logging training progress.")

    # Performance Arguments
    argument_parser.add_argument("--gpu", type=int,
                                 default=int(config_parser["performance"]["gpu"]),
                                 help="Integer to indicate which gpu to be used.")
    argument_parser.add_argument("--precision", type=int,
                                 default=int(config_parser["performance"]["precision"]),
                                 help="Integer for the level of precision should be used. 16 or 32 supported.")
    argument_parser.add_argument("--data_workers", type=int,
                                 default=int(config_parser["performance"]["data_workers"]),
                                 help="Integer for the number of data workers used to load the dataset.")

    # Model Arguments
    argument_parser.add_argument("--model_dir", type=str,
                                 default=config_parser["model"]["model_dir"],
                                 help="Directory where the model weights will be saved.")
    argument_parser.add_argument("--batch_size", type=int,
                                 default=int(config_parser["model"]["batch_size"]),
                                 help="Integer for the size of the batch used to train the model.")
    argument_parser.add_argument("--learning_rate", type=float,
                                 default=float(config_parser["model"]["learning_rate"]),
                                 help="Floating point value for the learning rate used to train the model.")
    argument_parser.add_argument("--efficient_net_b", type=int,
                                 default=int(config_parser["model"]["efficient_net_b"]),
                                 help="Integer for the compound coefficient for EfficientNet encoder.")

    # Dataset Arguments
    argument_parser.add_argument("--val_split", type=float,
                                 default=float(config_parser["dataset"]["val_split"]),
                                 help="Floating point value for determining the validation split.")
    argument_parser.add_argument("--image_size", type=int,
                                 default=int(config_parser["dataset"]["image_size"]),
                                 help="Integer for the dimension of the input images.")
    argument_parser.add_argument("--dataset_dir", type=str,
                                 default=config_parser["dataset"]["dataset_dir"],
                                 help="Directory path for the input dataset.")
    argument_parser.add_argument("--augmentation", type=str_to_bool,
                                 default=config_parser["dataset"]["augmentation"].lower() == "true",
                                 help="Boolean if augmentation should be used during training.")

    # Early Stopping Arguments
    argument_parser.add_argument("--window", type=int,
                                 default=int(config_parser["early_stopping"]["window"]),
                                 help="Integer fot the early stopping window.")
    argument_parser.add_argument("--target", type=float,
                                 default=float(config_parser["early_stopping"]["target"]),
                                 help="Floating point value for the target used for early stopping.")
    argument_parser.add_argument("--min_epochs", type=int,
                                 default=int(config_parser["early_stopping"]["min_epochs"]),
                                 help="Integer for the minimum number of epochs.")
    argument_parser.add_argument("--max_epochs", type=int,
                                 default=int(config_parser["early_stopping"]["max_epochs"]),
                                 help="Integer for the maximum number of epochs.")

    # Contrastive Predictive Coding Arguments
    argument_parser.add_argument("--cpc_alt_mask", type=str_to_bool,
                                 default=config_parser["cpc"]["cpc_alt_mask"].lower() == "true",
                                 help="Boolean if the alternative mask should be used.")
    argument_parser.add_argument("--cpc_code_size", type=int,
                                 default=int(config_parser["cpc"]["cpc_code_size"]),
                                 help="Integer for the size of the patch encodings.")
    argument_parser.add_argument("--cpc_patch_size", type=int,
                                 default=int(config_parser["cpc"]["cpc_patch_size"]),
                                 help="Integer for the size of the patches.")
    argument_parser.add_argument("--cpc_patch_stride", type=int,
                                 default=int(config_parser["cpc"]["cpc_patch_stride"]),
                                 help="Integer for the stride used for extracting patches.")
    argument_parser.add_argument("--cpc_random_patches", type=int,
                                 default=int(config_parser["cpc"]["cpc_random_patches"]),
                                 help="Integer for the number of random patches used in the loss function.")
    argument_parser.add_argument("--cpc_multi_directional", type=str_to_bool,
                                 default=config_parser["cpc"]["cpc_multi_directional"].lower() == "true",
                                 help="Boolean if the contrastive predictive coding should be multi directional.")

    # Debug Arguments
    argument_parser.add_argument("--batches_per_epoch", type=int,
                                 default=int(config_parser["debug"]["batches_per_epoch"]),
                                 help="Integer for the number of batches per epoch should be used.")

    # Returns the argument parser.
    arguments = argument_parser.parse_args()
    return vars(arguments)


def print_arguments(arguments):
    """
    Print all the arguments to the command line.
    :param arguments: ArgumentParser Namespace object.
    """

    # Cycles through all the arguments within the Namespace object.
    for key, value in arguments.items():
        log(arguments, f"{key: <24}: {value}")
