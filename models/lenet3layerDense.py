# lenet3layerDense.py
#
# This takes the LeNet reconstruction from:
# https://exowanderer.medium.com/what-is-this-keras-thing-anyways-fe7aa00158ef
#
# and lenet.py, and adds 1) the rescaling as a layer (just showing
# off); 2) Batch normalization; 3) a third layer; and 4) adds a second
# dense layer on the output.
#
# The third layer provides a minimalist version of the network here:
#
# commons.wikimedia.org/wiki/File:The-architecture-of-each-CNN-in-the-triplet-network.png
#
# while the second dense layer makes the output stage closer to the
# original output stage of LeNet as recorded here:
#
# https://pabloinsente.github.io/the-convolutional-network
#
# The extra layer on output distinguishes it from
# lenet3layer.py/LeNet3layer

from models.backbone import Backbone 
from tensorflow.keras import layers, models

class LeNet3layerDense(Backbone):
    # First we set up some constants that we will use to do this
    # across the various layers.
    kernel_shape = 3, 3    # train 3x3 kernels across all Conv layers
    activation = 'relu'    # use Rectified Linear Unit activiation functions
    pool_shape = 2, 2      # reduce dimensionality by 2 x 2 pooling
    dropout_rate = 0.5     # drop 50% of neurons
    padding = 'same'       # maintain the shape of feature maps per layer
    strides = 1            # do not downsample via stride

    nfilters_hidden1 = 32  # Start with 32 convolution filters to train
    nfilters_hidden2 = 64  # Then twice as many filters to train
    nfilters_hidden3 = 128 # Finish with a third block of 128 filters

    # Define how we will build the model
    model = models.Sequential(name='LeNet_3layer_Dense')

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
        # A batch normalization layer
        self.model.add(
            layers.BatchNormalization(
                name='First_Batch_Norm_Layer'
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
        # Another batch normalization layer. 
        self.model.add(
            layers.BatchNormalization(
                name='Second_Batch_Norm_Layer'
            )
        )
        # Reduce the dimensionality after the second Conv-layer w/ MaxPool2D
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
                name="Second_MaxPool2D_Layer"
            )
        )

        # Add the third convolution layer. This has 128 filters
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden3,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Third_Conv2D_Layer'
            )
        )
        # Another batch normalization layer. 
        self.model.add(
            layers.BatchNormalization(
                name='Third_Batch_Norm_Layer'
            )
        )
        # Reduce the dimensionality after the second Conv-layer w/ MaxPool2D
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
                name="Third_MaxPool2D_Layer"
            )
        )

        # Convert the 2D outputs to a 1-D vector in preparation for label prediction
        self.model.add(
            layers.Flatten(
                name="Flatten_from_Conv2D_to_Dense"
            )
        )

        #  Has two densely connected layers between convolution layers
        #  and the output.
        #
        # Dropout 50% of the neurons from the Conv+Flatten layers to regulate
        self.model.add(
            layers.Dropout(
                rate=self.dropout_rate,
                name="Dropout_from_Flatten_to_Dense"
            )
        )
        self.model.add(
            layers.Dense(
                units=120,
                activation=self.activation,
                name="First_Dense_layer"
            )
        )
        # Dropout 50% between Dense layers
        self.model.add(
            layers.Dropout(
                rate=self.dropout_rate,
                name="Dropout_from_Dense_to_Dense"
            )
        )
        self.model.add(
            layers.Dense(
                units=84,
                activation=self.activation,
                name="Second_Dense_layer"
            )
        )
        # Dropout 50% of the neurons from the Conv+Flatten layers to regulate
        self.model.add(
            layers.Dropout(
                rate=self.dropout_rate,
                name="Dropout_from_Dense_to_Output"
            )
        )

        # Compute the weighted-logistic for each possible label in one-hot encoding
        self.model.add(
            layers.Dense(
                units=self.num_classes,
                activation="softmax",
                name="n-Dimensional_Logistic_Output_Layer"
            )
        )
