# -*- coding: utf-8 -*-

"""
The file contains the implementations of the models used for the Multi-Directional Contrastive Predictive Coding.
    Encoder - Class for defining an encoder based on ResNeXt101.
    MaskedConv2D - Class to define a masked convolutional layer.
    MultiDirectionalPixelCNN - Class for defining a PixelCNN autoregressor with optional multi-directional.
"""


# Built-in/Generic Imports
import os

# Library Imports
import torch
import torch.nn as nn


__author__ = "Jacob Carse"
__copyright__ = "Copyright 2020, Multi-Directional Contrastive Predictive Coding for Histology"
__credits__ = ["Jacob Carse", "Stephen McKenna"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


class Encoder(nn.Module):
    """
    Class for the Encoder model, containing the methods:
        init - Initialiser for the model.
        forward_features - Forward propagation for the features of the encoder model.
        forward - Forward propagation for the encoder model.
        save_model - Method for saving the model.
    """

    def __init__(self, code_size, image_size=96, imagenet=False):
        """
        Initialiser for the model that initialises the models layers.
        :param code_size: The size of the output feature vectors.
        :param image_size: The size of each dimension of the input image.
        :param imagenet: If ImageNet weights should be used to initilise the model.
        """

        # Calls the super for the nn.Module.
        super(Encoder, self).__init__()

        # Loads the ResNeXt Model from PyTorch.
        self.model = torch.hub.load("pytorch/vision:v0.5.0", "resnext101_32x8d", pretrained=imagenet)

        # Gets the output dimensions of the encoder output.
        with torch.no_grad():
            temp_input = torch.zeros(1, 3, image_size, image_size)
            encoder_size = self.forward_features(temp_input).shape[1]

        # Initialises the code head for outputting feature vector.
        self.code_out = nn.Linear(encoder_size, code_size)

    def forward_features(self, x):
        """
        Forward propagates an input to output features for the ResNeXt encoder.
        :param x: PyTorch Tensor for the input image batch.
        :return: Feature vector output of the encoder.
        """

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        """
        Performs forward propagation with then encoder.
        :param x: PyTorch Tensor for the input image batch.
        :return: Feature vector equal to code size.
        """

        x = self.forward_features(x)
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


class Classifier(nn.Module):
    def __init__(self, feature_size, num_classes):
        """
        Initialiser for the model that initialises the models layers.
        :param feature_size: The size of the input feature vectors.
        :param num_classes: The number of classes the features can be classified as.
        """

        # Calls the super for the nn.Module.
        super(Classifier, self).__init__()

        # Hidden layer.
        self.liner = nn.Linear(feature_size, 512)

        # Output layer.
        self.out = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Performs forward propagation with then classifier.
        :param x: PyTorch Tensor for the input image batch.
        :return: Feature vector equal to code size.
        """

        x = self.liner(x)
        return self.out(x)

    def save_model(self, path, name, epoch=None):
        """
        Method for saving the classifier model.
        :param path: Directory path to save the classifier.
        :param name: The name of the experiment to be saved.
        :param epoch: Integer for the current epoch to be included in the save name.
        """

        # Checks if the save directory exists and if not creates it.
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # Saves the model to the save directory.
        if epoch is not None:
            torch.save(self.state_dict(), os.path.join(path, f"{name}_classifier_{str(epoch)}.pt"))
        else:
            torch.save(self.state_dict(), os.path.join(path, f"{name}_classifier.pt"))


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


class MultiDirectionalPixelCNN(nn.Module):
    """
    Class for the Multi-Directional Autoregressor model, containing the methods:
        init - Initiliser for the Multi-Directional PixelCNN.
        masked_block - Method to define a block with a Masked Convolutional layer.
        multi_directional_masked_block - Method to define a multi directional block with a Masked Convolutional layer.
        forward - Method for forward propagating the model.
        save_model - Method for saving the model.
    """

    def __init__(self, n_channels, h=128, multi_directional=True):
        """
        The initiliser for the Multi-Directional PixelCNN.
        :param n_channels: Integer for the number of input channels.
        :param h: Integer for the size fo the hidden layers in the PixelCNN.
        :param multi_directional: Boolean for if the Multi-Directional PixelCNN should be used.
        """

        # Calls the super for the nn.Module.
        super(MultiDirectionalPixelCNN, self).__init__()

        # Stores the boolean if multi directional pixel should be used.
        self.multi_directional = multi_directional

        # Defines the masked blocks for the PixelCNN
        self.conv_A = self.multi_directional_masked_block(n_channels, h, 'A')
        self.conv_B1 = self.multi_directional_masked_block(n_channels, h, 'B')
        self.conv_B2 = self.multi_directional_masked_block(n_channels, h, 'B')
        self.conv_B3 = self.multi_directional_masked_block(n_channels, h, 'B')
        self.conv_B4 = self.multi_directional_masked_block(n_channels, h, 'B')
        self.conv_B5 = self.multi_directional_masked_block(n_channels, h, 'B')

        # Defines the 1x1 Convolutional layers for the Multi-Directional PixelCNN.
        if multi_directional:
            self.convs = nn.ModuleList([nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)])
            for _ in range(4):
                self.convs.append(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0))

        # Defies the output block for the PixelCNN.
        self.out = nn.Sequential(nn.ReLU(),
                                 nn.Conv2d(1024 if multi_directional else 256, 1024, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(1024),
                                 nn.ReLU(),
                                 nn.Conv2d(1024, n_channels, kernel_size=1, stride=1, padding=0))

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
                             nn.BatchNorm2d(2 * h))

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
        if self.multi_directional:
            x = torch.cat([self.conv_A[0](x), self.conv_A[1](x).transpose(2, 3).flip(3),
                           self.conv_A[2](x).flip(2), self.conv_A[3](x).transpose(2, 3)], dim=1)
            x = self.convs[0](x)
            x = torch.cat([self.conv_B1[0](x), self.conv_B1[1](x).transpose(2, 3).flip(3),
                           self.conv_B1[2](x).flip(2), self.conv_B1[3](x).transpose(2, 3)], dim=1)
            x = self.convs[1](x)
            x = torch.cat([self.conv_B2[0](x), self.conv_B2[1](x).transpose(2, 3).flip(3),
                           self.conv_B2[2](x).flip(2), self.conv_B2[3](x).transpose(2, 3)], dim=1)
            x = self.convs[2](x)
            x = torch.cat([self.conv_B3[0](x), self.conv_B3[1](x).transpose(2, 3).flip(3),
                           self.conv_B3[2](x).flip(2), self.conv_B3[3](x).transpose(2, 3)], dim=1)
            x = self.convs[3](x)
            x = torch.cat([self.conv_B4[0](x), self.conv_B4[1](x).transpose(2, 3).flip(3),
                           self.conv_B4[2](x).flip(2), self.conv_B4[3](x).transpose(2, 3)], dim=1)
            x = self.convs[4](x)
            x = torch.cat([self.conv_B5[0](x), self.conv_B5[1](x).transpose(2, 3).flip(3),
                           self.conv_B5[2](x).flip(2), self.conv_B5[3](x).transpose(2, 3)], dim=1)

        # Forward propagation for PixelCNN.
        else:
            x = self.conv_A[0](x)
            x = self.conv_B1[0](x)
            x = self.conv_B2[0](x)
            x = self.conv_B3[0](x)
            x = self.conv_B4[0](x)
            x = self.conv_B5[0](x)

        # Output layer for both PixelCNN and Multi-Directional PixelCNN.
        x = self.out(x)

        # Returns the output reshaped to the original dimensions.
        return x.view(batch_size, c_in, height, width)

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
            torch.save(self.state_dict(), os.path.join(path, name + f"_autoregressor_{str(epoch)}.pt"))
        else:
            torch.save(self.state_dict(), os.path.join(path, name + "_autoregressor.pt"))
