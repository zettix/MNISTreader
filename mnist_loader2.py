#!/usr/bin/python3

import gzip
import numpy as np
from array import array
from PIL import Image, ImageDraw


"""Helper for MNIST files in IDX format.

  Based on "Do you really know how MNIST is stored?" at:
  https://medium.com/theconsole/do-you-really-know-how-mnist-is-stored-600d69455937
  and a pickle based loader at:
  https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py
  
  IDX files are stored as N-dimensional lists, e.g. [[[....][....]]]]
  But they are rectangular, as each dimension has a single length.
  
  There is a provision to flatten the last two dimensions, as neural
  networks prefer a flat array of numbers as opposed to a 2D array.
"""

DATA=[
  "MNIST/t10k-images-idx3-ubyte.gz",
  "MNIST/t10k-labels-idx1-ubyte.gz",
  "MNIST/train-images-idx3-ubyte.gz",
  "MNIST/train-labels-idx1-ubyte.gz",
]

DEBUG = True

def Debug(s):
  if DEBUG:
    print(s)

def load(fname):
  with gzip.open(fname, 'rb') as f:
    assert(b'\x00' == f.read(1))
    assert(b'\x00' == f.read(1))
    datatype = int.from_bytes(f.read(1), 'big')
    ndims = int.from_bytes(f.read(1), 'big')
    Debug(f"magic: datatype {datatype} ndims {ndims}")
    assert(datatype == 8)
    Debug(f"ndims: {ndims}")
    dims = []
    for d in range(ndims):
      dims.append(int.from_bytes(f.read(4), 'big'))
    for d in dims:
       Debug(f"d: {d}")
    return recdim(f, dims, 0)

def recdim(f, dims, level):
  """Recursively rebuild arbitrary size rectangular array,
  flattening the last two dimensions."""
  result = []
  num_to_read = dims[level]
  flatten = False
  if level == (len(dims) - 2) and len(dims) > 1:
    num_to_read *= dims[level + 1] 
    flatten = True
  for x in range(num_to_read):
    if level == len(dims) - 1 or flatten:
      result.append(int.from_bytes(f.read(1), 'big'))
    else:
      result.append(recdim(f, dims, level + 1))
  return result

def load_data_wrapper():
  """Wrapper to load all 4 MNIST and return 3-tuple of training/testing data.

  The original MNIST files have 10K and 60K entries respectively, so we
  take the last 10K off of the 60K file for validation and use the 10K
  files for tests.

  n.b. A class is not used here so we can discard our data when done.

  Original comments:
  --
  Return a tuple containing ``(training_data, validation_data,
  test_data)``. Based on ``load_data``, but the format is more
  convenient for use in our implementation of neural networks.

  In particular, ``training_data`` is a list containing 50,000
  2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
  containing the input image.  ``y`` is a 10-dimensional
  numpy.ndarray representing the unit vector corresponding to the
  correct digit for ``x``.

  ``validation_data`` and ``test_data`` are lists containing 10,000
  2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
  numpy.ndarry containing the input image, and ``y`` is the
  corresponding classification, i.e., the digit values (integers)
  corresponding to ``x``.

  Obviously, this means we're using slightly different formats for
  the training data and the validation / test data.  These formats
  turn out to be the most convenient for use in our neural network
  code.
  --

  Returns:
    (training_data, validataion_data, test_data)
  """
  data = {}

  for fname in DATA:
    data[fname] = load(fname)
    Debug(f"Loaded {fname}")

  test_inputs = [np.reshape(x, (784, 1)) for x in data[DATA[0]]]
  test_data = zip(test_inputs, data[DATA[1]])
  Debug(f"Done zip 10k")
  Debug(f"Loaded Labels 60k")
  train_labels = [vectorized_result(x) for x in data[ DATA[3]][:50000]   ]
  train_inputs = [np.reshape(x, (784, 1)) for x in data[ DATA[2]][:50000]     ]
  train_data = zip(train_inputs, train_labels)
  Debug(f"Zipped Training data.")
  validation_inputs = [np.reshape(x, (784, 1)) for x in data[ DATA[2]][:50000]     ]
  validation_data = zip(validation_inputs, data[DATA[3]][50000:])
  Debug(f"Zipped all data.")
  return (train_data, validation_data, test_data)

def vectorized_result(j):
  """Return a 10-dimensional unit vector with a 1.0 in the jth
  position and zeroes elsewhere.  This is used to convert a digit
  (0...9) into a corresponding desired output from the neural
  network."""
  e = np.zeros((10, 1))
  e[j] = 1.0
  return e

if __name__ == '__main__':
   load_data_wrapper()
