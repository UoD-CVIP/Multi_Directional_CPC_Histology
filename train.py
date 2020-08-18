# -*- coding: utf-8 -*-

"""
This file contains implementations of the functions used to train the multi directional CPC model:
    train_cpc - Function used to facilitate the training of the Multi Directional Contrastive Predictive Coding model.
"""


# Built-in/Generic Imports
import time

# Library Imports
import torch
from apex import amp
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Own Modules Imports
from utils import *
from dataset import *
from cpc_model import *


__author__ = "Jacob Carse"
__copyright__ = "Copyright 2020, Multi-Directional Contrastive Predictive Coding for Histology"
__credits__ = ["Jacob Carse", "Stephen McKenna"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


def train_cpc(arguments, device):
    """
    Function used to train the Multi-Directional Contrastive Predictive Coding model.
    :param arguments: Dictonary containing arguments.
    :param device: PyTorch device object.
    """

    # Loads a TensorBoard Summary Writer.
    if arguments["tensorboard"]:
        writer = SummaryWriter(os.path.join("TensorBoard", arguments["task"], arguments["experiment"]))

    # Loads the training data and splits it into a validation data.
    train_data = Dataset(arguments, "train")
    validation_data = train_data.get_validation_set()

    # Creates the data loaders for the training and validation data.
    training_data_loader = DataLoader(train_data, batch_size=arguments["batch_size"],
                                           shuffle=True, num_workers=arguments["data_workers"],
                                           pin_memory=False, drop_last=True)
    validation_data_loader = DataLoader(validation_data, batch_size=arguments["batch_size"],
                                          shuffle=False, num_workers=arguments["data_workers"],
                                          pin_memory=False, drop_last=True)

    # Shuffles the training and validation data.
    random_train_data = train_data.shuffle()
    random_validation_data = validation_data.shuffle()

    # Creates the random data loaders for the training and validation data.
    random_training_loader = DataLoader(random_train_data, batch_size=arguments["random_patches"],
                                             shuffle=True, num_workers=arguments["data_workers"],
                                             pin_memory=False, drop_last=True)

    random_validation_loader = DataLoader(random_validation_data, batch_size=arguments["random_patches"],
                                               shuffle=True, num_workers=arguments["data_workers"],
                                               pin_memory=False, drop_last=True)

    in_patches = int(((arguments["image_size"] + 2 * 0 - 1 * (arguments["cpc_patch_size"] - 1) - 1)
                      / arguments["cpc_patch_stride"]) + 1)

    log(arguments, "Loaded Datasets")

    # Initialises the encode and autoregressor.
    encoder = EfficientNetEncoder(arguments["efficient_net_b"], arguments["cpc_code_size"])
    autoregressor = MultiDirectionalPixelCNN(arguments["cpc_code_size"],
                                             multi_directional=arguments["cpc_multi_directional"])

    # Sets the models to training mode.
    encoder.train()
    autoregressor.train()

    # Moves the models to the selected device.
    encoder.to(device)
    autoregressor.to(device)

    # Combines the parameters from the two models.
    parameters = list(encoder.parameters()) + list(autoregressor.parameters())

    # Initialises a optimiser used to optimise the parameters of the models.
    optimiser = optim.Adam(params=parameters, lr=arguments["learning_rate"])

    # If 16 bit precision is being used change the model and optimisers precision.
    if arguments["precision"] == 16:
        [encoder, autoregressor], optimiser = amp.initialize([encoder, autoregressor], opt_lebel="O2", verbosity=False)

    # Checks if precision level is supported and if not defaults to 32.
    elif arguments["precision"] != 32:
        log(arguments, "Only 16 and 32 bit precision supported. Defaulting to 32 bit precision.")

    log(arguments, "Models Initialised")

    # Main logging variables declared.
    losses, val_losses = [], []
    best_loss, best_epoch, total_batches = 1e10, 0, 0
    start_time = time.time()

    log(arguments, "Training Timer Started\n")

    # The beginning of the main training loop.
    for epoch in range(1, arguments["cpc_max_epochs"] + 1):
        epoch_loss, num_batches = 0, 0

        # Loops through the dataset by each batch.
        for image, _ in training_data_loader:
            batch_losses = []

            # Loads the batch into memory and splits the batch into patches.
            image = image.to(device)
            image = extract_patches(arguments, image, in_patches)

            # Encodes the patches with the encoder.
            encoded_batch = encoder.forward(image)
            encoded_batch = encoded_batch.view(arguments["batch_size"], in_patches, in_patches, -1)
            encoded_batch = encoded_batch.permute(0, 3, 1, 2)

            # Loads the random patches into memory and splits into patches.
            random_batch, _ = next(iter(random_training_loader))
            random_batch = random_batch.to(device)
            random_batch = extract_patches(arguments, random_batch, in_patches)

            # Encodes the random patches with the encoder.
            random_encoded = encoder.forward(random_batch)
            random_encoded = random_encoded.view(arguments["cpc_random_patches"], in_patches, in_patches, -1)
            random_encoded = random_encoded.permute(0, 3, 1, 2)

            # Autoregressor predicts the encoded features of half the image from the other half.
            masked_batch = encoded_batch.clone()

            # Applies a mask to the encoded batch.
            if arguments["cpc_alt_mask"]:
                for i in range(1, 6):
                    for j in range(1, 6):
                        masked_batch[:, :, i, j] = 0
            else:
                for i in range(3, 7):
                    for j in range(0, 7):
                        masked_batch[:, :, i, j] = 0

            # Forward propagates the autoregressor with the masked batch.
            predictions = autoregressor.forward(masked_batch)

            # Loops through the images in the batch.
            for image in range(arguments["cpc_batch_size"]):

                # Gets the masked elements of the predicted and encoded patches.
                if arguments["cpc_alt_mask"]:
                    predicted_patches = predictions[image, :, 1:6, 1:6].reshape(1, -1)
                    target_patches = encoded_batch[image, :, 1:6, 1:6].reshape(1, -1)
                else:
                    predicted_patches = predictions[image, :, 3:7, 0:7].reshape(1, -1)
                    target_patches = encoded_batch[image, :, 3:7, 0:7].reshape(1, -1)

                # Calculates the dot terms for the predicted patches.
                good_dot_terms = torch.sum(predicted_patches * target_patches, dim=1)
                dot_terms = [torch.unsqueeze(good_dot_terms, dim=0)]

                # Loops through the random images for each batch.
                for random_image in range(arguments["cpc_random_patches"]):

                    # Gets the masked elements for the random patches.
                    if arguments["cpc_alt_mask"]:
                        random_patches = random_encoded[random_image, :, 1:6, 1:6].reshape(1, -1)
                    else:
                        random_patches = random_encoded[random_image, :, 3:7, 0:7].reshape(1, -1)

                    # Calculates the dot terms for the random patches.
                    bad_dot_terms = torch.sum(predicted_patches * random_patches, dim=1)
                    dot_terms.append(torch.unsqueeze(bad_dot_terms, dim=0))

                # Calculates the log softmax for all the dot terms.
                log_softmax = torch.log_softmax(torch.cat(dot_terms, dim=0), dim=0)
                batch_losses.append(-log_softmax[0, ])

            # Finds the loss of the batch by combinding the loss for each image in the batch.
            loss = torch.sum(torch.cat(batch_losses))

            # Backward propagates the loss over the model.
            if arguments["precision"] == 16:
                with amp.scale_loss(loss, optimiser) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Updates the weights of the optimiser using the back propagated loss.
            optimiser.step()
            optimiser.zero_grad()

            # Adds the batch loss to the epoch loss and updates the number of batches.
            epoch_loss += loss.item()
            num_batches += 1

            # Writes the batch loss to TensorBoard.
            if arguments["tensorboard"]:
                writer.add_scalar("Loss/batch", loss.item(), num_batches + total_batches)

            # Logs the details of the training batch.
            if num_batches % arguments["log_itervals"] == 0:
                log(arguments, "Time: {}s\tTrain Epoch: {} [{}/{}] ({:.0f}%)]\tLoss: {:.6f}".format(
                    str(int(time.time() - start_time)).rjust(6, '0'), str(epoch).rjust(2, '0'),
                    str(num_batches * arguments["cpc_batch_size"]).rjust(len(str(len(train_data))), '0'),
                    len(train_data), 100. * num_batches / len(train_data), epoch_loss / num_batches
                ))

            # Stops the epoch early if specified.
            if num_batches == arguments["batches_per_epoch"]:
                break

        # Writes the epoch loss to TensorBoard.
        if arguments["tensorboard"]:
            writer.add_scalar("Loss/train", epoch_loss / num_batches, epoch)

        # Validation
