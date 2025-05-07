# basic-vision

Some inital attempts at deep computer vision using Keras and TensorFlow.

This started as an exercise in providing code to talk about in a taught module that covered some basic deep learning where I wanted some simple models that would work on just a CPU (since the students on that module had a preference for woring on their personal laptops).

When the models at the simpler end of the spectrum ran reasonably fast on my laptop (without CUDA drivers installed) I decided to push the limits a bit, and found that on datasets like MNIST and CIFAR-10 I could even train my implementations of AlexNet and the lighter weight VGG networks in decent time (we're talking 3-4 minute per epoch for VGG11, and I was seeing convergence in less than 20 epochs).

Recalling Mark Liberman's breakfast experiment, and having an Easter break to hand, I decided to map out the performance of these simpler networks on MNIST, Fashion MNIST and CIFAR-10, just to get a sense of how performance varied.

Exploring the space between AlexNet and VGG11 on one hand, and various extensions of LeNet that added additional layers, I stumbled across the mysterious VGG8, which, somewhat like the Dread Pirate Roberts, is not a single entity, but a name that many have claimed (except the VGG group themselves). So naturally I implemented a number of the networks that go by that name, claiming no exhaustivity, while also riffing on the idea of an eight-weighted layer network myself.

Naturally this ended up taking a lot more than the Easter break to finish up...

Fuller details on all the models can be found in the README in the models folder.