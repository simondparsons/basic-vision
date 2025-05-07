# vgg8_2.py
#
# Simon Parsons
# 25-04-25
#
# This starts from the LeNet5 reconstruction from:
# https://exowanderer.medium.com/what-is-this-keras-thing-anyways-fe7aa00158ef
#
# and extends it to the description of what is called VGG8 here:
#
# Cai, Y., Tang, T., Xia, L., Li, B., Wang, Y. and Yang, H., 2019. Low
# bit-width convolutional neural network on RRAM. IEEE Transactions on
# Computer-Aided Design of Integrated Circuits and Systems, 39(7),
# pp.1414-1427.
#
# This is one of a number of alternative descriptions of networks
# going under the name VGG8. Note that it has twice as many filters at
# each stage than VGG11. It is unusual in VGG8 implementations in
# having a fourth convolutional layer.
#
# Note that my reimplementation adds batch normalization and dropout.

from models.backbone import Backbone 
from tensorflow.keras import layers, models

class VGG8_2(Backbone):
    # First we set up some constants that we will use to do this
    # across the various layers.
    kernel_shape = (3, 3)  # train 3x3 kernels across all Conv layers
    activation = 'relu'    # use Rectified Linear Unit activiation functions
    pool_shape = (2, 2)    # reduce dimensionality by 2 x 2 pooling
    dropout_rate = 0.5     # drop 50% of neurons
    padding = 'same'       # maintain the shape of feature maps per layer
    strides = 1            # do not downsample via stride

    # Filters in the convolution layers. 
    nfilters_hidden1 = 128  # Start with 128 convolution filters to train
    nfilters_hidden2 = 256  # Then twice as many filters to train
    nfilters_hidden3 = 512  # Then twice as many filters again. 
    nfilters_hidden4 = 1024  # Finish by doubling the number of filters once more.

    # Define how we will build the model
    model = models.Sequential(name='VGG8_2')

    def buildModel(self):
        # Create the input layer to understand the shape of each image and batch-size 
        self.model.add(
            layers.Input(
                shape=self.img_shape,
                # batch_size=batch_size,
                name='Image_Batch_Input_Layer',
            )
        )

        # Add a rescaling layer to convert the inputs to fall in the range (-1, 1).
        # https://machinelearningmastery.com/image-augmentation-with-keras-preprocessing-layers-and-tf-image/
        self.model.add(
            layers.Rescaling(
                1/127.5,
                offset=-1
            )
        )

        # Add the first pair of convolution layers. These have 128
        # filters
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden1,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Conv2D_Layer_11'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden1,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Conv2D_Layer_12'
            )
        )
        # A batch normalization layer
        self.model.add(
            layers.BatchNormalization(
                name='Batch_Norm_Layer_1'
            )
        )
        # Reduce the dimensionality after the first Conv-layer w/
        # MaxPool2D
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
                name="MaxPool2D_Layer_1"
            )
        )

        # Add the next convolution block. These two layers each have
        # 256 filters
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden2,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Conv2D_Layer_21'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden2,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Conv2D_Layer_22'
            )
        )
        # Another batch normalization layer. 
        self.model.add(
            layers.BatchNormalization(
                name='Batch_Norm_Layer_2'
            )
        )
        # Reduce the dimensionality after the second Conv-layer w/
        # MaxPool2D
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
                name="MaxPool2D_Layer_2"
            )
        )

        # Add the third convolution block. This has 2 convolution
        # layers, each with 512 filters.
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden3,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Conv2D_Layer_31'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden3,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Conv2D_Layer_32'
            )
        )
        
        # Another batch normalization layer. 
        self.model.add(
            layers.BatchNormalization(
                name='Batch_Norm_Layer_3'
            )
        )
        # Reduce the dimensionality after the third Conv-layer w/
        # MaxPool2D
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
               name="MaxPool2D_Layer_3"
            )
        )

        # Add the fourth convolution block, 1 layer, with 1024 filters.
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden4,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Conv2D_Layer_41'
            )
        )

        # Another batch normalization layer. 
        self.model.add(
            layers.BatchNormalization(
                name='Batch_Norm_Layer_4'
            )
        )
        # Reduce the dimensionality after the third Conv-layer w/
        # MaxPool2D
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
               name="MaxPool2D_Layer_4"
            )
        )

        # Convert the 2D outputs to a 1-D vector in preparation for
        # label prediction.

        self.model.add(
            layers.Flatten(
                name="Flatten_from_Conv2D_to_Dense"
            )
        )
        
        # Dropout 50% of the neurons from the Conv+Flatten layers to
        # regulate
        self.model.add(
            layers.Dropout(
                rate=self.dropout_rate,
                name="Dropout_from_Flatten_to_Dense"
            )
        )

        # The source paper describes a 1024-10 class layer here. I am
        # assuming that is the connection between the Flattening layer
        # (which will be 1024 in the work from that paper since the
        # output is 1x1) and the 10 classes output by this
        # layer. However an explicit 1024 unit layer would also be in
        # keeping with the VGG family architecture.
        
        # Compute the weighted-logistic for each possible label in
        # one-hot encoding
        self.model.add(
            layers.Dense(
                units=self.num_classes,
                activation="softmax",
                name="n-Dimensional_Logistic_Output_Layer"
            )
        )

        
