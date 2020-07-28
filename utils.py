# -*- coding: utf-8 -*-

"""
This file contains the following utility functions for the application:
    log - Function to both print and log a given input message.
"""


# Built-in/Generic Imports
import os


__author__ = "Jacob Carse"
__copyright__ = "Copyright 2020, Multi-Directional Contrastive Predictive Coding"
__credits__ = ["Jacob Carse"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


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
