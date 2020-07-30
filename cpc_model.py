# -*- coding: utf-8 -*-

"""

"""


# Built-in/Generic Imports
import os
import types

# Library Imports
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


__author__ = "Jacob Carse"
__copyright__ = "Copyright 2020, Multi-Directional Contrastive Predictive Coding"
__credits__ = ["Jacob Carse"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


class EfficientNetEncoder(nn.Module):
    """
    Class for the EfficientNet encoder model, containing the methods:
        init - Initialiser for the model.
        forward - Forward propagation for the encoder model.
        save_model - Method for saving the model.
    """

    def __init__(self, arguments):
        """
        Initialiser for the model that initialises hte models layers.
        :param arguments: Dictionary of arguments.
        """

        # Calls the super for the nn.Module.
        super(EfficientNetEncoder, self).__init__()

        # Gets the compound coefficient for EfficientNet.
        b = str(arguments['efficientnet_b'])

        # Loads the model with pretaining if selected.
        if arguments["pretraining"] == "imagenet":
            self.encoder = EfficientNet.from_pretrained(f"efficientnet-b{b}")
        else:
            self.encoder = EfficientNet.from_name(f"efficientnet-b{b}")

        # Pools the encoder outputs to a one dimensional array.
        self.encoder_pool = nn.AdaptiveAvgPool2d(1)

        # Gets the output dimensions of the encoder output.
        with torch.no_grad():
            temp_input = torch.zeros(1, 3, 96, 96)
            encoder_size = self.encoder.extract_features(temp_input).shape[1]

        # Initialises the code head for outputting feature vector.
        self.code_out = nn.Linear(encoder_size, arguments["code_size"])

    def forward(self, x):
        """
        Performs forward propagation with then EfficientNet encoder.
        :param x: PyTorch Tensor for the input image batch.
        :return: Feature vector output of the encoder.
        """

        # Performs feature extraction with the encoder model.
        x = self.encoder.extact_features(x)
        x = self.encoder_pool(x)
        x = x.view(x.shape[0], -1)
        return self.code_out(x)

    def save_model(self, path, name, epoch=None):
        """
        Method for saving the encoder model.
        :param path: Directory path to save the encoder.
        :param name: The name of the experiment to be saved.
        :param epoch: Integer for the current epoch to be included in the save name.
        """

        # Checks if the save directory exists and if not creates it.
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # Saves the model to the save directory.
        if epoch is not None:
            torch.save(self.state_dict(), os.path.join(path, f"{name}_encoder_{str(epoch)}.pt"))
        else:
            torch.save(self.state_dict(), os.path.join(path, f"{name}_encoder.pt"))


class MaskedConv2D(nn.Conv2d):
    """
    Convolutional Layer that applies a mask to the weights of the convolutional kernel, containing methods:
        init - Initiliser for the masked convolutional layer.
        forward - Method for forward propagation of the model.
    """

    def __init__(self, mask_type, rotation, c_in, c_out, k_size, stride, pad):
        """
        Initiliser for the masked convolutonal layer.
        :param mask_type: The type of the mask either A or B.
        :param rotation: Integer for the number of degrees the mask should be rotated.
        :param c_in: Integer for the number of input channels.
        :param c_out: Integer for the number of output channels.
        :param k_size: Integer for the size of the convolutional kernel.
        :param stride: Integer for the size of the convolution.
        :param pad: Integer for the amount of padding applied to each image.
        """

        # Calls the super for the nn.Conv2d.
        super(MaskedConv2D, self).__init__(c_in, c_out, k_size, stride, pad, bias=False)

        # Defines the size of the weights.
        ch_out, ch_in, height, width = self.weight.size()

        # Creates the weight mask.
        mask = torch.ones(ch_out, ch_in, height, width)

        # Alters the mask shape based on mask type.
        if mask_type == 'A':
            mask[:, :, height // 2, width // 2:] = 0
            mask[:, :, height // 2 + 1:] = 0
        elif mask_type == 'B':
            mask[:, :, height // 2, width // 2 + 1:] = 0
            mask[:, :, height // 2:] = 0

        # Rotates the mask to the given rotation.
        if rotation == 90:
            mask = mask.transpose(2, 3)
        elif rotation == 180:
            mask = mask.flip(2)
        elif rotation == 270:
            mask = mask.transpose(2, 3).flip(3)

        # Adds a persistent buffer for the mask.
        self.register_buffer("mask", mask)
        
    def forward(self, x):
        """
        The forward propagation for the masked convolutional layer.
        :param x: The input to the forward propagation pass.
        :return: The output of the forward propagation pass.
        """

        # Applies the mask to the weights of the convolutional layers.
        self.weight.data *= self.mask

        # Performs a forward pass with the masked weights.
        return super(MaskedConv2D, self).forward(x)


class MultiDirectionalPixelCnn(nn.Module):
    """

    """

    def __init__(self, n_channels, h=128, multi_directional=True):
        """
        The initiliser for the Multi-Directional PixelCNN.
        :param n_channels: Integer for the number of input channels.
        :param h: Integer for the size fo the hidden layers in the PixelCNN.
        :param multi_directional: Boolean for if the Multi-Directional PixelCNN should be used.
        """

        # Calls the super for the nn.Conv2d.
        super(MultiDirectionalPixelCnn, self).__init__()

        #
        self.multi_directional = multi_directional

    def masked_block(self, rotation, h):
        """
        A masked convolutional block using a given rotation.
        :param rotation: Integer for the rotation for the masked convolutional layer either 0, 90, 180 or 270.
        :param h: Integer for the size of the hidden layer.
        :return: PyTorch model with the masked convolutional block.
        """

        return nn.Sequential(nn.ReLU(),
                             nn.Conv2d(2 * h, h, 1),
                             nn.BatchNorm2d(h),
                             MaskedConv2D('B', rotation, h, h, 3, 1, 1),
                             nn.BatchNorm2d(h),
                             nn.ReLU(),
                             nn.Conv2d(h, 2 * h, 1),
                             nn.BatchNorm2d(2 * 2))

    def multi_directional_masked_block(self, n_channels, h, mask_type):
        """
        A multi dimensional masked block made of blocks from multiple dimensions.
        :param n_channels: Integer for the number of input channels.
        :param h: Integer for the size of the hidden layer.
        :param mask_type: The type of maksed block, A or B.
        :return: A ModuleList for the masked convolutional blocks.
        """

        # Defines a list of convolutional blocks.
        masked_conv = nn.ModuleList([])

        # Appends a block with 0 degree rotation.
        masked_conv.append(MaskedConv2D('A', 0, n_channels, 2 * h, k_size=7, stride=1,
                                        pad=3) if mask_type == 'A' else self.masked_block(0, h))

        # If multi-directional is being used append blocks with multiple directions
        if self.multi_directional:
            masked_conv.append(MaskedConv2D('A', 90, n_channels, 2 * h, k_size=7, stride=1,
                                            pad=3) if mask_type == 'A' else self.masked_block(90, h))
            masked_conv.append(MaskedConv2D('A', 180, n_channels, 2 * h, k_size=7, stride=1,
                                            pad=3) if mask_type == 'A' else self.masked_block(180, h))
            masked_conv.append(MaskedConv2D('A', 270, n_channels, 2 * h, k_size=7, stride=1,
                                            pad=3) if mask_type == 'A' else self.masked_block(270, h))

        # Returns the ModuleList containing masked convolutional blocks.
        return masked_conv

    def forward(self, x):
        """
        Forward propagation for the PixelCNN using multi-directional blocks if specified.
        :param x: The input to the forward propagation.
        :return: The output of the forward propagation.
        """

        # The dimensions of the input.
        batch_size, c_in, height, width = x.size()

        # Forward propagation for the multi-directional PixelCNN.
