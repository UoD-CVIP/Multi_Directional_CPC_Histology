# -*- coding: utf-8 -*-

"""
This file contains implementations of a class and a function to handle the dataset, containing:
    extract_patches - Function to extract patches from a tensor of image.
    Dataset - Class for the handling the dynamic loading and augmenting of images and labels.
"""


# Built-in/Generic Imports
import os

# PyTorch Imports
import torch
from torch.utils import data
from torchvision import transforms

# Library Imports
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


__author__ = "Jacob Carse"
__copyright__ = "Copyright 2020, Multi-Directional Contrastive Predictive Coding for Histology"
__credits__ = ["Jacob Carse", "Stephen McKenna"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


def extract_patches(arguments, images, num_patches):
    """
    Extracts a number of patches from a given tensor of images.
    :param arguments: Dictionary containing 'cpc_patch_size' and 'cpc_patch_stride'.
    :param images: A tensor of input images.
    :param num_patches: Integer for the number of patches to take from each dimension.
    :return: A tensor of patches equal to 'num_patches * num_patches'.
    """

    # List of all patches.
    all_patches = []

    # Cycles through all the inputted images.
    for i in range(images.shape[0]):

        # Cycles through the patches in the image.
        for y_patch in range(num_patches):
            for x_patch in range(num_patches):

                # Gets the y coordinates for the patch.
                y1 = y_patch * arguments["cpc_patch_stride"]
                y2 = y1 + arguments["cpc_patch_size"]

                # Gets the x coordinates for the patch.
                x1 = x_patch * arguments["cpc_patch_stride"]
                x2 = x1 + arguments["cpc_patch_size"]

                # Extracts the patch from the image.
                img_patch = images[i, :, y1:y2, x1:x2]
                img_patch = img_patch.view(images.shape[1], arguments["cpc_patch_size"], arguments["cpc_patch_size"])
                all_patches.append(img_patch)

    # Concatenates all the patches into a single tensor of images.
    all_patches = torch.cat(all_patches)
    return all_patches.view([-1, 3, arguments["cpc_patch_size"], arguments["cpc_patch_size"]])


class Dataset(data.Dataset):
    """
    Class for the handling and loading of images and labels.
        init - Initialiser for the class used to load an array of filenames and labels.
        len - Gets the length of the dataset.
        get_item - Gets a image and label from the dataset based on a given index.
        augment - Applies augmentations to the image and converts it to a tensor.
        get_validation_set - Splits the dataset into training and validation.
        reduce_size - Reduces the size of the training set.
    """

    def __init__(self, arguments, mode, filenames=None, labels=None):
        """
        Initialiser for the class that loads an array of filenames and labels for the dataset.
        :param arguments: Dictionary containing, 'dataset_dir', 'augmentation', 'float16' and 'validation_split'.
        :param mode: String specifying the mode of data to be loaded, 'train', 'validation' and 'test'.
        :param filenames: NumPy array of filenames, used if the mode is specified as validation.
        :param labels: NumPy array of labels, used if the mode is specified as validation.
        """

        # Calls the PyTorch Dataset initialiser.
        super(Dataset, self).__init__()

        # Stores the arguments and mode in the object.
        self.arguments = arguments
        self.mode = mode

        # Sets the validation data to the given input.
        if filenames is not None:
            self.filenames = filenames
            self.labels = labels

        # Loads the filenames and labels from a given dataset directory.
        else:
            # Loads the csv file containing the filenames and labels.
            csv_file = pd.read_csv(os.path.join(self.arguments["dataset_dir"], f"{mode}.csv"), names=['a', 'b'])

            # extracts the filenames and labels from the csv file.
            self.filenames = np.array(csv_file['a'].values.tolist())
            self.labels = np.array(csv_file['b'].values.tolist())

    def __len__(self):
        """
        Gets the length of the dataset.
        :return: Integer for the length of the dataset.
        """

        return len(self.filenames)

    def __getitem__(self, index):
        """
        Gets a given item from the dataset based on a given index.
        :param index: Integer representing the index to be extracted from the dataset.
        :return: A tensor containing the extracted image and a integer containing the corresponding label.
        """

        # Gets the filepath of the image.
        image_path = os.path.join(self.arguments["dataset_dir"], self.filenames[index])

        # Loads the image from the file path.
        image = Image.open(image_path)

        # Augments and converts the PIL image to a PyTorch Tensor.
        image = self.augment(image)

        # Returns the image tensor and label.
        if self.arguments["precision"] == 16:
            return image.type(torch.HalfTensor), int(self.labels[index])
        return image, int(self.labels[index])

    def augment(self, image):
        """
        Method for augmenting a given input image as a tensor.
        :param image: A PIL image.
        :return: A augmented image tensor.
        """

        # Mean and Standard Deviation for the PCAM dataset.
        mean = [0.7003911728173295, 0.5379628536502826, 0.6912184480517259]
        std = [0.18236434618801434, 0.20133812957358757, 0.16535855754524545]

        # Declare the standard transforms to the image tensor.
        transformations = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

        # Additional augmentations are applied is selected.
        if self.arguments["augmentation"]:
            # Random 90 degree rotations.
            class RandomRotation:
                def __init__(self, angles): self.angles = angles
                def __call__(self, x):
                    return transforms.functional.rotate(x, float(np.random.choice(self.angles)))

            # Additional transforms added to the list list of transformations.
            transformations = [transforms.RandomVerticalFlip(),
                               transforms.RandomHorizontalFlip(),
                               RandomRotation([0, 90, 180, 270])] + transformations

        # Return the image with the standard transforms applied.
        return transforms.Compose(transformations)(image)

    def get_validation_set(self):
        """
        Gets a Dataset object comprised of data extracted from the training set.
        :return: A Dataset object.
        """

        # Splits the filenames and labels into training and validation.
        self.filenames, val_filenames, self.labels, val_labels = train_test_split(self.filenames, self.labels,
                                                                                  test_size=self.arguments["val_split"])

        # Returns a Dataset object built using the validation data and labels.
        return Dataset(self.arguments, "validation", val_filenames, val_labels)

    def shuffle(self):
        """
        Shuffles the dataset and returns a new dataset with the shuffled data.
        :return: A dataset object with the shuffled data.
        """

        # Creates a random list with the indices of the data.
        indices = np.random.permutation(len(self.filenames))

        # Shuffles the filenames and labels.
        shuffled_data = self.filenames[indices]
        shuffled_labels = self.labels[indices]

        # Returns a Dataset object with the shuffled data.
        return Dataset(self.arguments, self.mode, shuffled_data, shuffled_labels)

    def reduce_size(self, reduction_size):
        """
        Reduces the size of the dataset.
        :param reduction_size: The dataset size to be reduced to.
        """

        # Checks if reduction_size is an acceptable size.
        if reduction_size > len(self.filenames) or reduction_size < 1:
            reduction_size = len(self.filenames)

        # Reduces the dataset.
        _, self.filenames, _, self.labels = train_test_split(self.filenames, self.labels,
                                                             test_size=reduction_size, stratify=self.labels)
