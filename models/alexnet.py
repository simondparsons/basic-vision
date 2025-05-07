# alexnet.py
#
# Simon Parsons
# 25-04-26
#
# This holds code for the AlexNet architecture as reported here:
#
# https://pabloinsente.github.io/the-convolutional-network
#
# with a couple of minor changes.
#
# Since AlexNet was developed for different sizes of image and number
# of classes than I have been playing with, I followed the general
# principle of 1) just using 3x3 kernels, 2) adjusting the strides in
# the convolution layers, 3) adjusting the strides and number pooling
# layers, and 4) the padding, to get a final image that was between
# 3x3 and 5x5.
#
# I have included the original Alexnet values for tehse parameters as
# comments in the Keras convolution layers so the code can be adjusted
# back to the original parameters for AlexNet easily. Or you can
# download the code at the URL above and tweak it to work with larger
# images.
#
# I also added batch normalization, following the convolutional
# layers. As I understand it, batch normalization post-dates AlexNet and
# would almost certainly have been used by its designers had it been
# available.
#
# TBC, I recreated the Keras description of AlexNet from the
# architecture diagram at the URL above, rather than copying the code
# provided. The code that is available there is hacked to work with
# CIFAR-10, but by jigging with the padding rather than changing
# filter size. Results don't seem to vary much.
#
# Note that looking at the discussion of AlexNet in
#
# K. Simonyan & A, Zisserman Very Deep Convolutional Networks for
# Large-scale Image Recognition, 3rd International Conference on
# Learning Representations, 2015
#
# which was the paper which introduced the VGG family, the larger
# filters in the early stages of AlexNet were the main distinguishing
# factor between it and VGG, so there is not much to choose between my
# code and that of the VGG family. For comparison this has 8 weighted
# layers so the comparison point is the VGG8 which, of course, is not
# featured in [Simonyam & Zisserman 2015].

from models.backbone import Backbone 
from tensorflow.keras import layers, models

class AlexNet(Backbone):
    # Here we set up some constants that we will use across the various layers.
    kernel_shape = (3, 3)# train 3x3 kernels across all Conv layers
    activation = 'relu'  # use Rectified Linear Unit activiation functions
    pool_shape = (2, 2)  # How to pool. Original Alexnet uses 3x3
    dropout_rate = 0.5   # drop 50% of neurons
    padding = 'same'     # maintain the shape of feature maps per layer
    strides = 1          # we keep the variable for continuity, but
                         # don't use it in the model since stride
                         # changes from layer to layer. 

    nfilters_hidden1 = 96  # Start with 96 convolution filters to train
    nfilters_hidden2 = 256 # then increase to 256
    nfilters_hidden3 = 384 # and finish with 384.

    # Define how we will build the model
    model = models.Sequential(name='AlexNet')

    def buildModel(self):
        # Create the input layer to understand the shape of each image and batch-size 
        self.model.add(
            layers.Input(
                shape=self.img_shape,
                # batch_size=batch_size,
                name='Image_Batch_Input_Layer',
            )
        )

        # Add the first convolution layer. This has 96 filters
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden1,
                kernel_size=self.kernel_shape,
                # kernel_size= (11, 11), # Original Alexnet value
                activation=self.activation,
                padding=self.padding,    # No padding for original Alexnet
                strides=1,
                # strides= (4, 4), # Original AlexNet value
                name='Conv2D_Layer_1'
            )
        )
        # A batch normalization layer
        self.model.add(
            layers.BatchNormalization(
                name='Batch_Norm_Layer_1'
            )
        )
        # Reduce the dimensionality after the first Conv-layer w/ MaxPool2D
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
                # pool_size = (3, 3), # Original AlexNet value
                # strides = (2, 2),   # Original AlexNet value
                name="MaxPool2D_Layer_1"
            )
        )

        # Add the next convolution layer. This has 256 filters
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden2,
                kernel_size=self.kernel_shape,
                # kernel_size= (5, 5), # Original AlexNet value
                activation=self.activation,
                padding=self.padding,  # No padding for original Alexnet
                strides=1,
                # strides= (2, 2), # Original AlexNet value
                name='Conv2D_Layer_2'
            )
        )
        # A batch normalization layer
        self.model.add(
            layers.BatchNormalization(
                name='Batch_Norm_Layer_2'
            )
        )
        # Reduce the dimensionality after the second Conv-layer w/ MaxPool2D
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
                # pool_size = (3, 3), # Original AlexNet value
                # strides = (2, 2),   # Original AlexNet value
                name="MaxPool2D_Layer_2"
            )
        )

        # Now 3 convolution layers with 384 filters each:
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden3,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                # padding=1, # Original Alexnet value
                strides=1,
                name='Conv2D_Layer_31'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden3,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                # padding=1, # Original Alexnet value
                strides=1,
                name='Conv2D_Layer_32'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden3,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                # padding=1, # Original Alexnet value
                strides=1,
                name='Conv2D_Layer_33'
            )
        )
        # A batch normalization layer
        self.model.add(
            layers.BatchNormalization(
                name='Batch_Norm_Layer_3'
            )
        )
        # Reduce the dimensionality after the second Conv-layer w/ MaxPool2D
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
                # pool_size = (3, 3), # Original AlexNet value
                # strides = (2, 2)    # Original AlexNet value
                name="MaxPool2D_Layer_3"
            )
        )
        
        # Convert the 2D outputs to a 1-D vector in preparation for label prediction
        self.model.add(
            layers.Flatten(
                name="Flatten_from_Conv2D_to_Dense"
            )
        )

        # Dropout 50% of the neurons from the Conv+Flatten layers to regulate
        self.model.add(
            layers.Dropout(
                rate=self.dropout_rate,
                name="Dropout_from_Flatten_to_Dense"
            )
        )
        # A fully connected layer. Original Alexnet has 4096 units but
        # it is classifying 1000 objects. For 10 classes, take our
        # inspiration from the description of LeNet at the same URL. 
        self.model.add(
            layers.Dense(
                units=120,
                # units = 4096, # Original AlexNet value
                activation="relu",
                name="Dense_layer_1"
            )
        )
        # Dropout 50% of the neurons between Dense layers. Original AlexNet 
        self.model.add(
            layers.Dropout(
                rate=self.dropout_rate,
                name="Dropout_from_Dense_to_Dense"
            )
        )
        # A fully connected layer. Original Alexnet has 4096 units.
        self.model.add(
            layers.Dense(
                units=84,
                # units = 4096, # Original AlexNet value
                activation="relu",
                name="Dense_Layer_2"
            )
        )
        # Dropout 50% of the neurons between Dense layers.
        self.model.add(
            layers.Dropout(
                rate=self.dropout_rate,
                name="Dropout_from_Dense_to_Output"
            )
        )
        
        # Compute the weighted-logistic for each possible label in
        # one-hot encoding
        self.model.add(
            layers.Dense(
                units=self.num_classes,
                activation="softmax",
                name="n-Dimensional_Logistic_Output_Layer"
            )
        )

