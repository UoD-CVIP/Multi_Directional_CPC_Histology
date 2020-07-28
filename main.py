#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
This file is the main executable for Multi-Directional Contrastive Predictive Coding for Histology.
This file loads the arguments, sets random seed and runs a selected task.
"""


# Own Module Import
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
