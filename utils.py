# -*- coding: utf-8 -*-

"""
This file contains the following utility functions for the application:
    str_to_bool - Function to convert a string to boolean value.
    log - Function to both print and log a given input message.
    set_random_seed - Function that sets a seed for all random number generation functions.
    get_device - Function to get the device that will be used.
    HDF5Handler - Class for the handling of HDF5 files.
"""


# Built-in/Generic Imports
import os
import random

# Library Imports
import h5py
import torch
import numpy as np
from argparse import ArgumentTypeError


__author__ = "Jacob Carse"
__copyright__ = "Copyright 2020, Multi-Directional Contrastive Predictive Coding for Histology"
__credits__ = ["Jacob Carse", "Stephen McKenna"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


def str_to_bool(argument):
    """
    Converts a string to a boolean value.
    :param argument: Input string.
    :return: Boolean value.
    """

    # Checks if the input is already a boolean.
    if isinstance(argument, bool):
        return argument

    # Checks if the input is True.
    elif argument.lower() == "true":
        return True

    # Checks if the input is False.
    elif argument.lower() == "false":
        return False

    # Returns an error if no boolean value was found.
    else:
        raise ArgumentTypeError("Boolean value expected.")


def log(arguments, message):
    """
    Logging function that will both print and log an input message.
    :param arguments: Dictionary containing 'log_dir' and 'experiment'.
    :param message: String containing the message to be printed.
    """

    # Prints the message.
    print(message)

    # Logs a message to a specified log directory is specified.
    if arguments["log_dir"] != '':

        # Creates a folder if one does not exist.
        os.makedirs(os.path.dirname(arguments["log_dir"]), exist_ok=True)

        # Logs te message to the log file.
        print(message, file=open(os.path.join(arguments["log_dir"], f"{arguments['experiment']}_log.txt"), 'a'))


def set_random_seed(seed):
    """
    Sets a random seed for each python library that generates random numbers.
    :param seed: Integer for the number used as the seed.
    """

    # Sets the random seed for python libraries.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Sets PyTorch to be deterministic if using CUDNN.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(arguments):
    """
    Sets the Default GPU that the device will use for training and testing.
    :param arguments: A Dictionary of arguments.
    :return: A PyTorch device.
    """

    # If GPU is set to less than 0 then set device to use CPU.
    if arguments["gpu"] < 0 and not torch.cuda.is_available():
        return torch.device("cpu")

    # Sets the GPU device to preferred device and checks if available.
    else:
        if arguments["gpu"] > torch.cuda.device_count() - 1:
            return torch.device(f"cuda:{torch.cuda.device_count() - 1}")
        else:
            return torch.device(f"cuda:{arguments['gpu']}")


class HDF5Handler(object):
    """
    A class for handling of a HDF5 file including, creation and data appending.
        init - Initialiser for the handler and also creates the HDF5 file.
        append - Append values from a one dimensional array.
    """

    def __init__(self, data_path, data_set, shape, data_type=np.float32, compression="gzip", chunk_len=1):
        """
        Initialiser for the HDF5 handler which creates the file with the given arguments.
        :param data_path: String for the file path where the HDF5 file will be stored.
        :param data_set: The initial data set for the HDF5 file.
        :param shape: The shape of the HDF5 file.
        :param data_type: The data type of the HDF5 file.
        :param compression: The compression used for the HDF5 file.
        :param chunk_len: The chunk length of the HDF5 file.
        """

        # Saved the inital parameters into the handler object.
        self.data_path = data_path
        self.data_set = data_set
        self.shape = shape
        self.i = 0

        # Creates the HDF5 file.
        with h5py.File(self.data_path, mode='w') as h5_file:
            self.data = h5_file.create_dataset(data_set,
                                               shape=(0, ) + shape,
                                               maxshape=(None, ) + shape,
                                               dtype=data_type,
                                               compression=compression,
                                               chunks=(chunk_len, ) + shape)

    def append(self, values):
        """
        Appends an array of values to a HDF5 file.
        :param values: A list type object of values to be added to the HDF5 file.
        """

        # Loads the HDF5 file in append mode.
        with h5py.File(self.data_path, mode='a') as h5_file:

            # Gets the data from the HDF5 file.
            data = h5_file[self.data_set]

            # Realizes the HDF5 data.
            data.resize((self.i + len(values), ) + self.shape)

            # Adds the values to the data.
            data[self.i: self.i + len(values)] = values

            # Increases the length of the HDF5 data.
            self.i += len(values)

            # Flush the HDF5 file's buffers.
            h5_file.flush()
