# MNISTreader
Read MNIST Database reader for use with neural-networks-and-deep-learning or other uses.

This is for python3.   The network.py file is also updated for python3.
One can then follow the excellent book on neural networks and run
the training code using this site:
<http://neuralnetworksanddeeplearning.com/chap1.html>

This loader has made possible using:
"Do you really know how MNIST is stored?":
<https://medium.com/theconsole/do-you-really-know-how-mnist-is-stored-600d69455937>
and a pickle based loader at:
<https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py>
Which did not funtion for me using python3, hence this replacement MNIST loader.  
Usage is simple:
`import mnist_loader2 as ml2`
`training_data, validation_data, test_data = ml2.load_data_wrapper()`

And then build your network, train and test.  Enjoy!
