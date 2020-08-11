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
    argument_parser.add_argument("--experiment", type=str,
                                 default=config_parser["standard"]["experiment"],
                                 help="String representing the name of the current experiment.")
    argument_parser.add_argument("--task", type=str,
                                 default=config_parser["standard"]["task"],
                                 help="String representing the task for the application to run.")

    # Logging Arguments
    argument_parser.add_argument("--log_dir", type=str,
                                 default=config_parser["logging"]["log_dir"],
                                 help="Directory where the log files will be stored.")

    # Performance Arguments
    argument_parser.add_argument("--gpu", type=int,
                                 default=int(config_parser["performance"]["gpu"]),
                                 help="Integer to indicate which gpu to be used.")

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
