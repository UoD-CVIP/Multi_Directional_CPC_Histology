## -*- coding: utf-8 -*-

"""
This file contains implementations of the functions used to train a CNN model:
    train_cnn - Function used to facilitate the training of the Convolutinal Neural Network model.
    test_cnn - Function used for the testing of training of the Convolutinal Neural Network model.
"""


# Built-in/Generic Imports
import os
import time

# Library Imports
import torch
from torch import optim
from torch.cuda import amp
from torch.nn import functional as F
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


def train_cnn(arguments, device):
    """
    Function used to train the Contrastive Predictive Coding Model.
    :param arguments: Dictionary containing arguments.
    :param device: PyTorch device object.
    :return: Returns lists of training and validation losses and an integer for the best performing epoch.
    """

    # Loads a TensorBoard Summary Writer.
    if arguments["tensorboard"]:
        writer = SummaryWriter(os.path.join("TensorBoard", arguments["task"], arguments["experiment"]))

    # Loads the training data and reduces its size.
    train_data = Dataset(arguments, "train")
    train_data.reduce_size(arguments["training_examples"])

    # Splits the training data into a validation set.
    validation_data = train_data.get_validation_set()

    # Creates the data loaders for the training and validation data.
    training_data_loader = DataLoader(train_data, batch_size=arguments["batch_size"],
                                           shuffle=True, num_workers=arguments["data_workers"],
                                           pin_memory=False, drop_last=False)
    validation_data_loader = DataLoader(validation_data, batch_size=arguments["batch_size"],
                                          shuffle=False, num_workers=arguments["data_workers"],
                                          pin_memory=False, drop_last=False)

    log(arguments, "Loaded Datasets")

    # Initialises the encoder and autoregressor.
    encoder = Encoder(arguments["cpc_code_size"], arguments["image_size"],
                      imagenet=arguments["pretrained"].lower() == "imagenet")
    classifier = Classifier(encoder.encoder_size, 2, arguments["hidden_layer"])

    # Sets the models to training mode.
    encoder.train()
    classifier.train()

    # Moves the models to the selected device.
    encoder.to(device)
    classifier.to(device)

    # Loads weights from pretrained Contrastive Predictive Coding model.
    if arguments["pretrained"].lower() == "cpc":
        encoder_path = os.path.join(arguments["model_dir"], f"{arguments['experiment']}_encoder_best.pt")
        encoder.load_state_dict(torch.load(encoder_path, map_location=device), strict=False)

    # Combines the parameters from the two models.
    parameters = list(encoder.parameters()) + list(classifier.parameters())

    # Initialises a optimiser used to optimise the parameters of the models.
    optimiser = optim.Adam(params=parameters, lr=arguments["learning_rate"])

    # If 16 bit precision is being used change the model and optimiser precision.
    if arguments["precision"] == 16:
        scaler = amp.GradScaler()

    # Checks if precision level is supported and if not defaults to 32.
    elif arguments["precision"] != 32:
        log(arguments, "Only 16 and 32 bit precision supported. Defaulting to 32 bit precision.")

    log(arguments, "Models Initialised")

    # Main logging variables declared.
    start_time = time.time()
    losses, validation_losses = [], []
    best_loss, best_epoch, total_batches = 1e10, 0, 0

    log(arguments, "Training Timer Started\n")

    # The beginning of the main training loop.
    for epoch in range(1, arguments["max_epochs"] + 1):
        epoch_acc, epoch_loss, num_batches = 0, 0, 0

        # Loops through the dataset by each batch.
        for images, labels in training_data_loader:

            # Loads the batch into memory.
            images = images.to(device)
            labels = labels.to(device)
            
            if arguments["precision"] == 16:
                with amp.autocast():
                    # Encodes the images with the encoder.
                    encoded_images = encoder.forward_features(images)

                    # Classifies the encoded images.
                    predictions = classifier.forward(encoded_images)

                    # Calculates the batch accuracy.
                    batch_acc = (predictions.max(dim=1)[1] == labels).sum().double() / labels.shape[0]

                    # Finds the loss of the batch using the predictions.
                    loss = F.cross_entropy(predictions, labels.type(torch.long))
                    
                scaler.scale(loss).backward()
                    
                scaler.step(optimiser)
                    
                scaler.update()

                optimiser.zero_grad()
                
            else:
                # Encodes the images with the encoder.
                encoded_images = encoder.forward_features(images)

                # Classifies the encoded images.
                predictions = classifier.forward(encoded_images)

                # Calculates the batch accuracy.
                batch_acc = (predictions.max(dim=1)[1] == labels).sum().double() / labels.shape[0]

                # Finds the loss of the batch using the predictions.
                loss = F.cross_entropy(predictions, labels.type(torch.long))
                
                loss.backward
                optimiser.step()
                optimiser.zero_grad()
                

            # Adds the batch loss to the epoch loss and updates the number of batches.
            num_batches += 1
            epoch_acc += batch_acc
            epoch_loss += loss.item()

            # Writes the batch loss to TensorBoard
            if arguments["tensorboard"]:
                writer.add_scalar("Loss/batch", loss.item(), num_batches + total_batches)
                writer.add_scalar("Accuracy/batch", batch_acc, num_batches + total_batches)

            # Logs the details of the training batch.
            if num_batches % arguments["log_intervals"] == 0:
                log(arguments, "Time: {}s\tTrain Epoch: {} [{}/{}] ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}".format(
                    str(int(time.time() - start_time)).rjust(6, '0'), str(epoch).rjust(2, '0'),
                    str(num_batches * arguments["batch_size"]).rjust(len(str(len(train_data))), '0'),
                    len(train_data), 100. * num_batches / (len(train_data) / arguments["batch_size"]),
                    epoch_loss / num_batches, epoch_acc / num_batches
                ))

            # Stops the epoch early if specified.
            if num_batches == arguments["batches_per_epoch"]:
                break

        # Adds the number of batches to total number.
        total_batches += num_batches

        # Writes the epoch loss to TensorBoard.
        if arguments["tensorboard"]:
            writer.add_scalar("Loss/train", epoch_loss / num_batches, epoch)
            writer.add_scalar("Accuracy/train", epoch_acc / num_batches, epoch)

        # Performs a validation epoch with no gradients.
        with torch.no_grad():
            # Logging metrics for validation epoch.
            validation_acc, validation_loss, validation_batches = 0, 0, 0

            for images, labels in validation_data_loader:
                # Loads the batch into memory.
                images = images.to(device)
                labels = labels.to(device)
                
                if arguments["precision"] == 16:
                    with amp.autocast():
                        # Encodes the images with the encoder.
                        encoded_images = encoder.forward_features(images)

                        # Classifies the encoded images.
                        predictions = classifier.forward(encoded_images)
                        
                else:
                    # Encodes the images with the encoder.
                    encoded_images = encoder.forward_features(images)

                    # Classifies the encoded images.
                    predictions = classifier.forward(encoded_images)

                # Calculates the batch accuracy.
                batch_acc = (predictions.max(dim=1)[1] == labels).sum().double() / labels.shape[0]

                # Finds the loss of the batch using the predictions.
                loss = F.cross_entropy(predictions, labels.type(torch.long))

                # Adds the batch loss and accuracy to the epoch loss and accuracy and updates the number of batches.
                validation_batches += 1
                validation_loss += loss.item()
                validation_acc += batch_acc

                # Stops the epoch early if specified.
                if num_batches == arguments["batches_per_epoch"]:
                    break

        # Writes the epoch loss to TensorBoard.
        if arguments["tensorboard"]:
            writer.add_scalar("Loss/validation", validation_loss / validation_batches, epoch)
            writer.add_scalar("Accuracy/validation", validation_acc / validation_batches, epoch)

        # Adds the training and validation losses to lists of losses.
        losses.append(epoch_loss / num_batches)
        validation_losses.append(validation_loss / validation_batches)

        # Logs the epoch information.
        log(arguments, "\nEpoch: {}\Training Loss: {:.6f}\tTraining Accuracy: {:.6f}\t"
                       "Validation Loss: {:.6f}\tValidation Accuracy: {:.6f}\n\n".
            format(epoch, losses[-1], epoch_acc / num_batches,
                   validation_losses[-1], validation_acc / validation_batches))

        # Checks if the validation loss is the best achieved loss and saves the model.
        if validation_losses[-1] < best_loss:
            best_loss = validation_losses[-1]
            best_epoch = epoch
            encoder.save_model(arguments["model_dir"], arguments["experiment"], "cnn_best")
            classifier.save_model(arguments["model_dir"], arguments["experiment"], "cnn_best")

        # Saves the models with reference to the current epoch.
        #encoder.save_model(arguments["model_dir"], arguments["experiment"], f"cnn_{epoch}")
        #classifier.save_model(arguments["model_dir"], arguments["experiment"], f"cnn_{epoch}")

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
    log(arguments, f"\n\nTraining finished after {epoch} epochs in {int(time.time() - start_time)}s")

    # Returns the loss values from the training.
    return losses, validation_losses, best_epoch


def test_cnn(arguments, device):
    """
    Function used to test a trained Convolutional Neural Network model.
    :param arguments: Dictionary containing arguments.
    :param device: PyTorch device object.
    :return: Returns the loss for the testing data on the model.
    """

    # Loads the testing dataset.
    test_data = Dataset(arguments, "test")

    # Creates the data loader for the testing data.
    test_data_loader = DataLoader(test_data, batch_size=arguments["batch_size"],
                                          shuffle=False, num_workers=arguments["data_workers"],
                                          pin_memory=False, drop_last=False)

    log(arguments, "Loaded Testing Data")

    # Initialises the encoder and loads the trained weights.
    encoder = Encoder(arguments["cpc_code_size"], arguments["image_size"])
    encoder_path = os.path.join(arguments["model_dir"], f"{arguments['experiment']}_encoder_cnn_best.pt")
    encoder.load_state_dict(torch.load(encoder_path, map_location=device), strict=False)

    # Initialises the classifier and loads the trained weights.
    classifier = Classifier(encoder.encoder_size, 2, arguments["hidden_layer"])
    classifier_path = os.path.join(arguments["model_dir"], f"{arguments['experiment']}_classifier_cnn_best.pt")
    classifier.load_state_dict(torch.load(classifier_path, map_location=device), strict=False)

    # Sets the models to evaluation mode.
    encoder.eval()
    classifier.eval()

    # Moves the models to the selected device.
    encoder.to(device)
    classifier.to(device)

    log(arguments, "Models Initialised")

    # Performs a testing epoch with no gradients.
    with torch.no_grad():
        # Logging metrics for the testing epoch.
        loss, accuracy, num_batches = 0, 0, 0

        # Loops through the testing dataset.
        for images, labels in test_data_loader:

            # Loads the batch into memory.
            images = images.to(device)
            labels = labels.to(device)

            if arguments["precision"] == 16:
                with amp.autocast():
                    # Encodes the images with the encoder.
                    encoded_images = encoder.forward_features(images)

                    # Classifiers the encoded images.
                    predictions = classifier.forward(encoded_images)
            else:
                # Encodes the images with the encoder.
                encoded_images = encoder.forward_features(images)

                # Classifiers the encoded images.
                predictions = classifier.forward(encoded_images)

            # Calculates the batch accuracy.
            accuracy += (predictions.max(dim=1)[1] == labels).sum().double() / labels.shape[0]

            # Finds the loss of the batch using the predictions.
            loss += F.cross_entropy(predictions, labels.type(torch.long)).item()

            # Updates the number of batches.
            num_batches += 1

            # Stops the epoch early if specified.
            if num_batches == arguments["batches_per_epoch"]:
                break

    # Calculates the loss and accuracy of the epoch.
    loss /= num_batches
    accuracy /= num_batches

    # Logs and returns the testing outputs.
    log(arguments, "\nTesting Loss: {:.6f}\tTesting Accuracy: {:.6f}".format(loss, accuracy))
    return loss, accuracy
