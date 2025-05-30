# vgg9.py
#
# Simon Parsons
# 25-04-25
#
# This started from the LeNet5 reconstruction from:
# https://exowanderer.medium.com/what-is-this-keras-thing-anyways-fe7aa00158ef
#
# and extended it to the description of what is called VGG8 here:
#
# Zhang, Q., Shen, Y. and Yi, Z., 2019, November. Video-based traffic
# sign detection and recognition. In 2019 International Conference on
# Image and Video Processing, and Artificial Intelligence (Vol. 11321,
# pp. 284-291). SPIE.
#
# making the model I have called VGG8_3, but some sloppy editing added
# another convolutional block with a single 128 filter layer. That
# would be unremarkable,e xcept that it seemed to outperform other
# models in the VGG8 family (possibly just because it is slightly
# deeper). Anyhow, reader I kept it and called it VGG9.

from models.backbone import Backbone 
from tensorflow.keras import layers, models

class VGG9(Backbone):
    # First we set up some constants that we will use to do this
    # across the various layers.
    kernel_shape = 3, 3    # train 3x3 kernels across all Conv layers
    activation = 'relu'    # use Rectified Linear Unit activiation functions
    pool_shape = 2, 2      # reduce dimensionality by 2 x 2 pooling
    dropout_rate = 0.5     # drop 50% of neurons
    padding = 'same'       # maintain the shape of feature maps per layer
    strides = 1            # do not downsample via stride

    # Filters in the convolution layers. 
    nfilters_hidden1 = 32  # Start with 32 convolution filters to train
    nfilters_hidden2 = 64  # Then twice as many filters to train
    nfilters_hidden3 = 128 # Then twice as many filters again. 

    # Define how we will build the model
    model = models.Sequential(name='VGG9')

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

        # Add the fourth convolution block, 1 layer, with 128 filters
        # (including this layer was the mistake).
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden3,
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
        #
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

        # The source paper says nothing about the FC layers, so I have
        # just filled in with the same as in VGG8_1.
        self.model.add(
            layers.Dense(
                units=1024,
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
        
        # Compute the weighted-logistic for each possible label in
        # one-hot encoding
        self.model.add(
            layers.Dense(
                units=self.num_classes,
                activation="softmax",
                name="n-Dimensional_Logistic_Output_Layer"
            )
        )        

        
