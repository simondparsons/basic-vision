# CNN Models

The sources of the architecture is recorded in each file, but they are also listed here for clarity. All the code grew out of this helpful article:

https://exowanderer.medium.com/what-is-this-keras-thing-anyways-fe7aa00158ef

but diverged as learnt more about Keras and early CNN architectures.

These started as simple CNNs that made sense from a teaching perspective, where I wanted something that was both easy to explain from scratch, and could be run on very limited hardware, as the students in question like to use their laptops rather than the fancy lab computers we provide. When the simplest models proved easy and quick to run on my laptop without CUDA, I started exploring the space of ore complex models from the early days of CNNs

## LeNet Variations

### LeNet

A modern reconstruction from:
https://exowanderer.medium.com/what-is-this-keras-thing-anyways-fe7aa00158ef
which uses 3x3 filters, max-pooling after convolutional layers, and a simplified output stage.

For a more authentic version see:
https://pabloinsente.github.io/the-convolutional-network

I stuck with the first since I wanted an example to teach from, and it makes sense for that to be a more modern style. To summarize it has one 32 filter convolutional layer and one 64 filter convolutional layer, with a single 10 unit output layer.

### LeNet 3 Layer

An extension of LeNet with a third convolutional layer. This gives us a 32 filter convolutional layer, a 64 filter layer and a 128 filter layer with the same 10 unit output layer as LeNet. There is batch normalization after the convolutional layers.

### LeNet 3 Layer Dense

To experiment with the effect of more FC output layers, I created a version of the 3 layer LeNet with two FC layers between the last convolutional layer and the 10 unit output stage. So we have: a 32 filter convolutional layer, a 64 filter layer and a 128 filter layer (with max pooling and batch norm), a 120 unit FC layer, an 84 unit FC layer and then the 10 unit output layer.

### LeNet Plus

This bulks up the number of layers towards a VGG-type model. We have 2 x 32 filter convolutional layers, 2 x 64 filter layers and 3 x 128 filter layers with the same 10 unit output layer as LeNet. As throughout, when I say m x n filters, it denotes a block of m layers each with n filters that is followed by batch normalization and max pooling.

For comparison with models below, it has 8 layers, but 7 of these are convolutional.

### LeNet Plus Dense

LeNet Plus with the larger number of FC layers. We have 2 x 32 filter convolutional layers, 2 x 64 filter layers and 3 x 128 filter layers then a 120 unit FC layer, an 84 unit FC layer and then the 10 unit output layer.

A total of 9 layers, with one more FC layer than LeNet Plus.

## VGG8 Variations

VGG8 is not a specific architecture from [Simonyan and Zisserman 2015] (despite what some folk say) since the simplest architecture in that paper is VGG11, with 11 weighted layers. Instead a number of people have created their own models that they call VGG8.

I have implemented:

### VGG8_1

From:
Zhang, K., Jia, X., Cao, K., Wang, J., Zhang, Y., Lin, K., Chen, L., Feng, X., Zheng, Z., Zhang, Z. and Zhang, Y., 2022. High On/Off Ratio Spintronic Multi‐Level Memory Unit for Deep Neural Network (Adv. Sci. 13/2022). Advanced Science, 9(13), p.2270086.

3 sets of two convolutional layers, 128, 256 and 512 filters in each, followed by a 1024 unit FC layer and an n-class output layer.

### VGG8_2

From:
Cai, Y., Tang, T., Xia, L., Li, B., Wang, Y. and Yang, H., 2019. Low bit-width convolutional neural network on RRAM. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 39(7), pp.1414-1427.

3 sets of two convolutional layers, 128, 256 and 512 filters in each, followed by a convolutional layer with 1024 filters, and an n-class output layer.

### VGG8_3

From:
Zhang, Q., Shen, Y. and Yi, Z., 2019, November. Video-based traffic sign detection and recognition. In 2019 International Conference on Image and Video Processing, and Artificial Intelligence (Vol. 11321, pp. 284-291). SPIE.

3 sets of pairs of convolutional layers, with 32, 64 and 128 filters, followed by a 1024 unit FC layer and an n-class output layer.

This configuration has one less convolution layer than LeNetPlusDense (one with 128 filters) but has a Dense layer with more units.

### VGG8_4

Note that none of the above have the same number of filters at each stage as VGG11, so there is an obvious architecture to explore which has pairs of layers with 64, 128 and 256 filters followed by a 1024 unit FC layer and an n-class output layer. (In other words, it is VGG8_3 with double the number of filters in the convolutional stages.)

I would argue this is closest to VGG11 since it has the same pattern of filters from the input layer as all the VGG models in [Simonyan and Zisserman 2015] (it is basically the first six convolution layers of VGG13). But see below.

### VGG8_5

From:
Thorat, P., Tongaonkar, R. and Jagtap, V., 2020. Towards designing the best model for classification of fish species using deep neural networks. In Proceeding of International Conference on Computational Science and Applications: ICCSA 2019 (pp. 343-351). Springer Singapore.

An 8 layer VGG variant with: a convolution layer with 32 filters, two convolution layers with 64 filters, 3 convolution layers with 128 filters, followed by a 1024 unit FC layer and an n-class output layer. (So VGG8_3 replacing a 32 filter layer with a 128 filter layer.) As they point out this is the first 6 layers of VGG11 (though with the number of filters halved).

### VGG8_F/M

Probably the most authentic since the come from the VGG group, but (as with VGG16 etc) the paper that introdcued them didn't given them the VGG name, just calling them the Fast and Medium architectures, but they have the requisite 8 layers. (There is also a Slow architecture, which differs only in strides and using 3x3 pooling --- I wanted to focus on a common pooling and filtering regime and so ignored this one.)

* Fast: a 64 filter layer, a 256 filter layer, then 3 x 256 filter layers followed by 2 FC layers and an output layer.

* Medium: a 96 filter layer, a 256 filter layer, then 3 x 512 filter layers followed by 2 FC layers and an output layer.

Interestingly the first pairs of layers use 11x11 and 5x5 filters (Slow) or 7x7 and 5x5 (Medium), so harking back to AlexNet there, and ramping up the number of filters faster than VGG11 and so on do.

Also note that, unlike the VGG8_ns above, which have 6 convolution layers and 2 other weighted layers, these both have 5 convolutional layers and 3 other weighted layers.

## AlexNet

### Original AlexNet

This take its structure from:
https://pabloinsente.github.io/the-convolutional-network

but is modified to use 3x3 filters so that it works for the 28x28 and 32x32 images in MNIST Fashion MNIST and CIFAR-10. The structure is: a 96 filter convolution layer, a 256 filter convolution layer, 3 x 384 filter convolution layers, two FC layers and an output layer. The FC layers follow the "dense" pattern of a 120 unit FC layer, an 84 unit FC layer and then the 10 unit output layer that I borrowed from LeNet.

This has 8 layers in total, so is a good comparison point for the VGG8 family, thoughmaybe not so good as the following variation.

### AlexNet Plus

Replaces the output layers of the above AlexNet reconstruction (which came from LeNet) with those from VGG8_F/M. This creates an archiecture that has the same combinations of layers, but in terms of number of filters sits in between them.

## VGG9

A mistake. The initial implementation of VGG8_3 ended up with a 128 unit convolution layer just before the FC layer. By the time I discovered the mistake I had run all the experiments and found this interloper had outperformed other VGG8 variants on Fashion MNIST. TBC, the architecture was 3 sets of pairs of convolutional layers, with 32, 64 and 128 filters, followed by a convolutional layer with 128 filters, a 1024 unit FC layer and an n-class output layer. With the additional layer it is a very lightweight 9 layer network.

## VGG11

The lightest model from:
K. Simonyan & A, Zisserman Very Deep Convolutional Networks for Large-scale Image Recognition, 3rd International Conference on Learning Representations, 2015.

but one that is by far the deepest in number of layers of this set of models (and a good end point for something run on a laptop). We have a 64 filter layer, a 128 filter layer, 2 x 256 filter layers, 2 x 512 filter layers and another 2 x 512 filter layers, followed by the usual 120 unit FC layer, an 84 unit FC layer and then the 10 unit output layer.

## Deeper networks

VGG11 could be easily modified to give the popular VGG16 and VGG19 architectures from [Simonyan & Zisserman 2015], but these didn't meet my needs for teaching where most studets want to run code on their laptops.