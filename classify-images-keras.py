# classify-images-keras.py
#
# Simon Parsons
# 25-04-25
#
# Code to explore the performance of early CNN architectures on
# standard problems from the Keras datasets.
#
# Code largely from:
# https://exowanderer.medium.com/what-is-this-keras-thing-anyways-fe7aa00158ef
#
# but modified to call different networks and to use command line
# arguments, allowing easy experimentation with different
# architectures and datasets.
#
# All of these architectures can be run in reasonable time on a modern
# CPU, meaning training on a single epoch on any of the datasets takes
# no more than ten minutes. (And they rarely need to train for more
# than 20 epochs on these datasets).

import string
import argparse
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, callbacks, utils, datasets, models
from operator import itemgetter
# These are the architectures available roughly in order of complexity.
from models.lenet import LeNet
from models.lenet3layer import LeNet3layer
from models.lenet3layerDense import LeNet3layerDense
from models.lenetPlus import LeNetPlus
from models.lenetPlusDense import LeNetPlusDense
from models.vgg8_1 import VGG8_1
from models.vgg8_2 import VGG8_2
from models.vgg8_3 import VGG8_3
from models.vgg8_4 import VGG8_4
from models.vgg8_5 import VGG8_5
from models.vgg8_11 import VGG8_11
from models.vgg8_F import VGG8_F
from models.vgg8_M import VGG8_M
from models.vgg9 import VGG9
from models.vgg11 import VGG11
from models.alexnet import AlexNet
from models.alexnetPlus import AlexNetPlus

def main():
    # Generalise the code by allowing the model, dataset and some of
    # the hyperparameters to be picked on the command line.
    parser = argparse.ArgumentParser(description='Keras/TensorFlow for image classification')
    # Possible values: mnist, fashion_mnist, cifar10
    parser.add_argument('--dataset', help='Which datset to use.', default='mnist')
    # Possible values: many, see the README.md in the models folder
    parser.add_argument('--model', help='Which model to use.', default='LeNet')
    # Possible values: y or yes for display, anothing else for no display
    parser.add_argument('--display', help='Display training data?', default='n')
    # Use epochs to specify a number to run without early stopping. If
    # you don't specify the script will run 50 epochs with early
    # stopping (which for the 3 simple datsets has rarely been more
    # than 20).
    parser.add_argument('--epochs', help='Specify number of epochs')
    # Batch size, in case we need to adjust this
    parser.add_argument('--batch_size', help='Specify batch size', default=64)
    # Patience, in case we need to adjust this
    parser.add_argument('--patience', help='How many epochs to wait before inviking early stopping ', default=3)

    args = parser.parse_args()

    # Load the data from Keras. Choice of MNIST, Fashion MNIST and CIFAR-10
    #
    dataset = args.dataset
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    elif dataset == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()
    elif dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
    else:
        print("I don't know the dataset:", args.dataset)
        exit(0)
    
    # Conv2D, the main Keras model layer we will use, requires 4D
    # inputs: batch, row, col, color. If we have no color dimension
    # (as in MNIST), add the color dimensions to represent greyscale.
    if np.ndim(X_train) == 3: 
        COLOR_DIM = -1
        X_train = np.expand_dims(X_train, axis=COLOR_DIM)
        X_test = np.expand_dims(X_test, axis=COLOR_DIM)

    # Pull out key features of the data. This assumes that the first image
    # is the same size as the rest.
    num_classes = np.unique(y_train).__len__()  
    img_shape = X_train[0].shape
    print("Classes:", num_classes)
    print("Image dimensions:", img_shape)

    # One-hot encode the output
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    # For flexibility, we will separate out the NN definition so that we
    # can experiment with a range of backbones.

    # Input the model description
    arch = args.model

    if arch == 'LeNet':
        network = LeNet(img_shape, num_classes)
    elif arch == 'LeNet3layer':
        network = LeNet3layer(img_shape, num_classes)
    elif arch == 'LeNet3layerDense':
        network = LeNet3layerDense(img_shape, num_classes)
    elif arch == 'LeNetPlus':
        network = LeNetPlus(img_shape, num_classes)
    elif arch == 'LeNetPlusDense':
        network = LeNetPlusDense(img_shape, num_classes)
    elif arch == 'VGG8_1':
        network = VGG8_1(img_shape, num_classes)
    elif arch == 'VGG8_2':
        network = VGG8_2(img_shape, num_classes)
    elif arch == 'VGG8_3':
        network = VGG8_3(img_shape, num_classes)
    elif arch == 'VGG8_4':
        network = VGG8_4(img_shape, num_classes)
    elif arch == 'VGG8_5':
        network = VGG8_5(img_shape, num_classes)
    elif arch == 'VGG8_11':
        network = VGG8_11(img_shape, num_classes)
    elif arch == 'VGG8_F':
        network = VGG8_F(img_shape, num_classes)
    elif arch == 'VGG8_M':
        network = VGG8_M(img_shape, num_classes)
    elif arch == 'VGG9':
        network = VGG9(img_shape, num_classes)
    elif arch == 'VGG11':
        network = VGG11(img_shape, num_classes)
    elif arch == 'AlexNet':
        network = AlexNet(img_shape, num_classes)
    elif arch ==  'AlexNetPlus':
        network = AlexNetPlus(img_shape, num_classes)
    else:
        print("I don't know the model:", args.model)
        exit(0)
    
    network.buildModel()
    print(network.model.name)
    model = network.model

    # Now compile the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    # Train the model.
    #
    # epochs:           How many iterations should we cycle over
    #                   the entire MNIST dataset
    # validation_split: How many images to hold out per epoch
    # batch size:       Could be 32, 64, 128, 256, 512
    # early_stopping:   When to stop training if performance plateaus.
    
    validation_split = 0.1  
    batch_size = 64 # The larger the batch size, the more memory a
                    # given dataset uses.

    # If we have specified the number of epochs, then run for that
    # number irrespective of the way that training goes. Otherwise do
    # early stopping after validation error hadn't improved for
    # patience=3 epochs.
    if args.epochs:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=batch_size,
            epochs=int(args.epochs),
            # The alternative is to explicitly set validation_data 
            validation_split = validation_split,
        )
    else:
        early_stopping = callbacks.EarlyStopping(patience=int(args.patience))
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=batch_size,
            epochs=50,
            validation_split = validation_split,
            callbacks=[early_stopping]
        )    

    # Print a summary of the model
    model.summary()

    # Show the change in accuracy and loss over training.
    if args.display == 'y' or args.display == 'yes':
        epochs = np.arange(len(history.history['loss']))

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15,10))
        ax1.plot(epochs, history.history['loss'], 'o-', label='Training Loss')
        ax1.plot(epochs, history.history['val_loss'], 'o-', label='Validation Loss')
        ax1.legend()
    
        ax2.plot(epochs, history.history['accuracy'], 'o-', label='Training Accuracy')
        ax2.plot(epochs, history.history['val_accuracy'], 'o-', label='Validation Accuracy')
        ax2.legend()

        plt.show()

    test_score = model.evaluate(X_test, y_test, verbose=0)
    train_score = model.evaluate(X_train, y_train, verbose=0)

    print("Train loss     :", train_score[0])
    print("Train accuracy :", train_score[1])
    print()
    print("Test loss      :", test_score[0])
    print("Test accuracy  :", test_score[1])

    return 0

if __name__ == "__main__":
    main()
