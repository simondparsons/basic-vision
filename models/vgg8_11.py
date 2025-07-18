# vgg8_11.py
#
# Simon Parsons
# 25-04-28
#
# Oddly, in all the menagerie of VGG8 papers, there is no network
# which exactly copies the convolutional layers of VGG11. Until now.
# This starts from the original VGG paper:
#
# K. Simonyan & A, Zisserman Very Deep Convolutional Networks for
# Large-scale Image Recognition, 3rd International Conference on
# Learning Representations, 2015
#
# where VGG11 is the least complex/deep of the models. That had 8
# convolutional layers and the 3 fully connected (Dense in Keras
# terms) layers. This has the first 5 of those convolutional layers,
# and then the same 3 fully connected layers as my VGG11
# implementation (see that model for why they are like they are).
#
# Note that there is no batch normalization in the original VGG11.

from models.backbone import Backbone 
from tensorflow.keras import layers, models

class VGG8_11(Backbone):
    # First we set up some constants that we will use across the
    # various layers.
    kernel_shape = (3, 3)  # train 3x3 kernels across all Conv layers
    activation = 'relu'    # use Rectified Linear Unit activiation functions
    pool_shape = (2, 2)    # reduce dimensionality by 2 x 2 pooling
    dropout_rate = 0.5     # drop 50% of neurons
    padding = 'same'       # maintain the shape of feature maps per layer
    strides = 1            # do not downsample via stride

    # Filters in the convolution layers.
    nfilters_hidden1 = 64   # Start with 64 convolution filters to train
    nfilters_hidden2 = 128  # Then twice as many filters to train
    nfilters_hidden3 = 256  # The doubling the number of filters once more.
    nfilters_hidden4 = 512  # The final layers all have 512 filters

    # Define how we will build the model
    model = models.Sequential(name='VGG8_11')

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

        # Add the first convolution layer with 64 filters
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

        # Add the next convolution block, again 1 layer this time with
        # 128 filters
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
        # layers, each with 256 filters.
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

        # Add the fourth convolution block. This has 1 convolution
        # with 512 filters, completing the 5 convolutional layers of
        # this cut down VGG11.
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

        # Reduce the dimensionality after the fourth Conv-layer w/
        # MaxPool2D
        #self.model.add(
        #    layers.MaxPooling2D(
        #        pool_size=self.pool_shape,
        #       name="MaxPool2D_Layer_4"
        #    )
        #)


        # Convert the 2D outputs to a 1-D vector in preparation for
        # label prediction
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
        # The output stage in all the VGG models had 4096, 4096 and
        # 1000 units to predict 1000 classes. Following VGG_F and
        # VGG_M, we use 1024 here.
        self.model.add(
            layers.Dense(
                units=1024,
                activation=self.activation,
                name="Dense_Layer_1"
            )
        )
        # Dropout 50% between Dense layers
        self.model.add(
            layers.Dropout(
                rate=self.dropout_rate,
                name="Dropout_from_Dense_to_Dense_1"
            )
        )
        self.model.add(
            layers.Dense(
                units=1024,
                activation=self.activation,
                name="Dense_Layer_2"
            )
        )
        # Dropout 50% between Dense layers
        self.model.add(
            layers.Dropout(
                rate=self.dropout_rate,
                name="Dropout_from_Dense_to_Dense_2"
            )
        )
        # Compute the weighted-logistic for each possible label in
        # one-hot encoding
        self.model.add(
            layers.Dense(
                units=self.num_classes, #10 classes in MNIST etc
                activation="softmax",
                name="n-Dimensional_Logistic_Output_Layer"
            )
        )
