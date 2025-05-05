# lenet.py
#
# Simon Parsons
# 25-04-25
#
# This holds the LeNet reconstruction from:
# https://exowanderer.medium.com/what-is-this-keras-thing-anyways-fe7aa00158ef
#
# and the structure is confirmed here:
#
# https://pabloinsente.github.io/the-convolutional-network
#
# though the latter has many fewer filters at each convolution, but an additional
# dense layer at the end.

from models.backbone import Backbone 
from tensorflow.keras import layers, models

class LeNet(Backbone):
    # Here we set up some constants that we will use to do this across
    # the various layers.
    kernel_shape = 3, 3  # train 3x3 kernels across all Conv layers
    activation = 'relu'  # use Rectified Linear Unit activiation functions
    pool_shape = 2, 2    # reduce dimensionality by 2 x 2 pooling
    dropout_rate = 0.5   # drop 50% of neurons
    padding = 'same'     # maintain the shape of feature maps per layer
    strides = 1          # do not downsample via stride

    nfilters_hidden1 = 32  # Start with 32 convolution filters to train
    nfilters_hidden2 = 64  # end with twice as many filters to train next

    # Define how we will build the model
    model = models.Sequential(name='LeNet_Reconstruction')

    def buildModel(self):
        # Create the input layer to understand the shape of each image
        # and batch-size
        self.model.add(
            layers.Input(
                shape=self.img_shape,
                # batch_size=batch_size,
                name='Image_Batch_Input_Layer',
            )
        )

        # Add the first convolution layer. This has 32 filters
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden1,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='First_Conv2D_Layer'
            )
        )

        # Reduce the dimensionality after the first Conv-layer w/ MaxPool2D
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
                name="First_MaxPool2D_Layer"
            )
        )

        # Add the next convolution layer. This has 64 filters
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden2,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Second_Conv2D_Layer'
            )
        )

        # Reduce the dimensionality after the second Conv-layer w/ MaxPool2D
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
                name="Second_MaxPool2D_Layer"
            )
        )

        # Convert the 2D outputs to a 1-D vector in preparation for
        # label prediction
        self.model.add(
            layers.Flatten(
                name="Flatten_from_Conv2D_to_Dense"
            )
        )

        # Dropout 50% of the neurons from the Conv+Flatten layers to regulate
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

