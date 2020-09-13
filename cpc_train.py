## -*- coding: utf-8 -*-

"""
This file contains implementations of the functions used to train the multi directional CPC model:
    train_cpc - Function used to facilitate the training of the Multi Directional Contrastive Predictive Coding model.
    test_cpc - Function used for the testing of training Multi Directional Contrastive Predictive Coding model.
"""


# Built-in/Generic Imports
import os
import time

# Library Imports
import torch
from apex import amp
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Own Modules Imports
from utils import *
from models import *
from dataset import *


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
    :param arguments: Dictionary containing arguments.
    :param device: PyTorch device object.
    :return: Returns lists of training and validation losses and an integer for the best performing epoch.
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
    random_training_loader = DataLoader(random_train_data, batch_size=arguments["cpc_random_patches"],
                                             shuffle=True, num_workers=arguments["data_workers"],
                                             pin_memory=False, drop_last=True)

    random_validation_loader = DataLoader(random_validation_data, batch_size=arguments["cpc_random_patches"],
                                               shuffle=True, num_workers=arguments["data_workers"],
                                               pin_memory=False, drop_last=True)

    # Calculates the number of patches across each dimension.
    in_patches = int(((arguments["image_size"] + 2 * 0 - 1 * (arguments["cpc_patch_size"] - 1) - 1)
                      / arguments["cpc_patch_stride"]) + 1)

    log(arguments, "Loaded Datasets")

    # Initialises the encoder and autoregressor.
    encoder = Encoder(arguments["cpc_code_size"], arguments["image_size"])
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
        [encoder, autoregressor], optimiser = amp.initialize([encoder, autoregressor], optimiser,
                                                             opt_level="O2", verbosity=False)

    # Checks if precision level is supported and if not defaults to 32.
    elif arguments["precision"] != 32:
        log(arguments, "Only 16 and 32 bit precision supported. Defaulting to 32 bit precision.")

    log(arguments, "Models Initialised")

    # Main logging variables declared.
    losses, validation_losses = [], []
    best_loss, best_epoch, total_batches = 1e10, 0, 0
    start_time = time.time()

    log(arguments, "Training Timer Started\n")

    # The beginning of the main training loop.
    for epoch in range(1, arguments["max_epochs"] + 1):
        epoch_loss, num_batches = 0, 0

        # Loops through the dataset by each batch.
        for batch, _ in training_data_loader:
            batch_losses = []

            # Loads the batch into memory and splits the batch into patches.
            batch = batch.to(device)
            batch = extract_patches(arguments, batch, in_patches)

            # Encodes the patches with the encoder.
            encoded_batch = encoder.forward(batch)
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

            # Clones the encoded batch for masking.
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
            for image in range(arguments["batch_size"]):

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
            loss = torch.mean(torch.cat(batch_losses))

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
            if num_batches % arguments["log_intervals"] == 0:
                log(arguments, "Time: {}s\tTrain Epoch: {} [{}/{}] ({:.0f}%)]\tLoss: {:.6f}".format(
                    str(int(time.time() - start_time)).rjust(6, '0'), str(epoch).rjust(2, '0'),
                    str(num_batches * arguments["batch_size"]).rjust(len(str(len(train_data))), '0'),
                    len(train_data), 100. * num_batches / (len(train_data) / arguments["batch_size"]),
                    epoch_loss / num_batches
                ))

            # Stops the epoch early if specified.
            if num_batches == arguments["batches_per_epoch"]:
                break

        # Writes the epoch loss to TensorBoard.
        if arguments["tensorboard"]:
            writer.add_scalar("Loss/train", epoch_loss / num_batches, epoch)

        # Performs a validation epoch with no gradients.
        with torch.no_grad():

            # Logging metrics for validation epoch.
            validation_loss, validation_batches = 0, 0

            # Loops through the validation dataset.
            for batch, _ in validation_data_loader:
                batch_losses = []

                # Moves the batch to the selected device and splits the images to patches.
                batch = batch.to(device)
                batch = extract_patches(arguments, batch, in_patches)

                # Encodes the patches with the encoder.
                encoded_batch = encoder.forward(batch)
                encoded_batch = encoded_batch.view(arguments["batch_size"], in_patches, in_patches, -1)
                encoded_batch = encoded_batch.permute(0, 3, 1, 2)

                # Loads the random patches into memory and splits into patches.
                random_batch, _ = next(iter(random_validation_loader))
                random_batch = random_batch.to(device)
                random_batch = extract_patches(arguments, random_batch, in_patches)

                # Encodes the random patches with the encoder.
                random_encoded = encoder.forward(random_batch)
                random_encoded = random_encoded.view(arguments["cpc_random_patches"], in_patches, in_patches, -1)
                random_encoded = random_encoded.permute(0, 3, 1, 2)

                # Clones the encoded batch for masking.
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
                for image in range(arguments["batch_size"]):

                    # Gets the masked elements for the random patches.
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
                    batch_losses.append(-log_softmax[0,])

                # Combines the loss for each image into a batch loss
                validation_loss += torch.mean(torch.cat(batch_losses))
                validation_batches += 1

                # Stops the epoch early if specified.
                if validation_batches == arguments["batches_per_epoch"]:
                    break

        # Writes the validation loss to TensorBoard
        if arguments["tensorboard"]:
            writer.add_scalar("Loss/validation", validation_loss / validation_batches, epoch)

        # Adds the epoch and validation loss to a list of losses.
        losses.append(epoch_loss / num_batches)
        validation_losses.append(validation_loss / validation_batches)

        # Logs the epoch information.
        log(arguments, "\nEpoch: {}\tLoss{:.6f}\tValidation Loss: {:.6f}\n\n".
            format(epoch, losses[-1], validation_losses[-1]))

        # Adds the total number of batches.
        total_batches += num_batches

        # Checks if the validation loss is the best achieved loss and saves the model.
        if validation_losses[-1] < best_loss:
            best_loss = validation_losses[-1]
            best_epoch = epoch
            encoder.save_model(arguments["model_dir"], arguments["experiment"], "best")
            autoregressor.save_model(arguments["model_dir"], arguments["experiment"], "best")

        # Saves the models with reference to the current epoch
        encoder.save_model(arguments["model_dir"], arguments["experiment"], epoch)
        autoregressor.save_model(arguments["model_dir"], arguments["experiment"], epoch)

        # Checks if training has performed the minimum number of epochs.
        if epoch >= arguments["min_epochs"]:

            # Calculates the generalised validation loss.
            g_loss = 100 * ((validation_losses[-1] / min(validation_losses[:-1])) - 1)

            # Calculates the training progress using a window over the training losses.
            t_progress = 1000 * ((sum(losses[-(arguments["window"] + 1): - 1]) /
                                  (arguments["window"] * min(losses[-(arguments["window"] + 1): - 1]))) - 1)

            # Compares the generalised loss and training progress against a selected target value.
            if g_loss / t_progress > arguments["target"]:
                break

    # Logs that the training has finished.
    log(arguments, f"\n\nTraining Finished after {epoch} epochs in {int(time.time() - start_time)}s")

    # Returns the loss values from the training.
    return losses, validation_losses, best_epoch


def test_cpc(arguments, device):
    """
    Function used to test a trained Multi-Directional Contrastive Predictive Coding model.
    :param arguments: Dictionary containing arguments.
    :param device: PyTorch device object.
    :return: Returns the loss for the testing data on the model.
    """

    # Loads the testing dataset.
    test_data = Dataset(arguments, "test")

    # Creates the data loader for the testing data.
    test_data_loader = DataLoader(test_data, batch_size=arguments["batch_size"],
                                          shuffle=False, num_workers=arguments["data_workers"],
                                          pin_memory=False, drop_last=True)

    # Shuffles a copy of the testing data.
    random_test_data = test_data.shuffle()

    # Creates the data loader for the random testing data.
    random_test_loader = DataLoader(random_test_data, batch_size=arguments["cpc_random_patches"],
                                    shuffle=True, num_workers=arguments["data_workers"],
                                    pin_memory=False, drop_last=True)

    # Calculates the number of patches across each dimension.
    in_patches = int(((arguments["image_size"] + 2 * 0 - 1 * (arguments["cpc_patch_size"] - 1) - 1)
                      / arguments["cpc_patch_stride"]) + 1)

    log(arguments, "Loaded Testing Data")

    # Initialises the encoder and autoregressor.
    encoder = Encoder(arguments["cpc_code_size"], arguments["image_size"])
    autoregressor = MultiDirectionalPixelCNN(arguments["cpc_code_size"],
                                             multi_directional=arguments["cpc_multi_directional"])

    # Loads the trained weights of the encoder and autoregressor.
    encoder.load_state_dict(torch.load(os.path.join(arguments["model_dir"], arguments["experiment"] +
                                                    "_encoder_best.pt"), map_location=device))
    autoregressor.load_state_dict(torch.load(os.path.join(arguments["model_dir"], arguments["experiment"] +
                                                          "_autoregressor_best.pt"), map_location=device))

    # Sets the models to evaluation mode.
    encoder.eval()
    autoregressor.eval()

    # Moves the models to the selected device.
    encoder.to(device)
    autoregressor.to(device)

    # If 16 bit precision is being used change the model and optimisers precision.
    if arguments["precision"] == 16:
        [encoder, autoregressor] = amp.initialize([encoder, autoregressor], opt_level="O2", verbosity=False)

    # Checks if precision level is supported and if not defaults to 32.
    elif arguments["precision"] != 32:
        log(arguments, "Only 16 and 32 bit precision supported. Defaulting to 32 bit precision.")

    log(arguments, "Models Initialised")

    # Performs a testing epoch with no gradients.
    with torch.no_grad():

        # Logging metrics for the testing epoch.
        loss, num_batches = 0, 0

        # Loops through the testing dataset.
        for batch, _ in test_data_loader:
            batch_losses = []

            # Moves the batch to the selected device and splits the images to patches.
            batch = batch.to(device)
            batch = extract_patches(arguments, batch, in_patches)

            # Encodes the patches with the encoder.
            encoded_batch = encoder.forward(batch)
            encoded_batch = encoded_batch.view(arguments["batch_size"], in_patches, in_patches, -1)
            encoded_batch = encoded_batch.permute(0, 3, 1, 2)

            # Loads the random patches into memory and splits into patches.
            random_batch, _ = next(iter(random_test_loader))
            random_batch = random_batch.to(device)
            random_batch = extract_patches(arguments, random_batch, in_patches)

            # Encodes the random patches with the encoder.
            random_encoded = encoder.forward(random_batch)
            random_encoded = random_encoded.view(arguments["cpc_random_patches"], in_patches, in_patches, -1)
            random_encoded = random_encoded.permute(0, 3, 1, 2)

            # Clones the encoded batch for masking.
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
            for image in range(arguments["batch_size"]):

                # Gets the masked elements for the random patches.
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
                batch_losses.append(-log_softmax[0,])

            # Combines the loss for each image into a batch loss.
            loss += torch.mean(torch.cat(batch_losses))
            num_batches += 1

            # Stops the epoch early if specified.
            if num_batches == arguments["batches_per_epoch"]:
                break

    # Gets the testing loss from the batch losses.
    loss /= num_batches
    log(arguments, "\nTesting Loss: {:.6f}".format(loss))

    # Returns the testing loss.
    return loss
