#!/usr/bin/env python
from __future__ import print_function
import sys

for p in sys.path:
    if 'egg' in p and 'theanets' in p:
        sys.path.remove(p)
reload(sys)

import urllib


def print_process(a, b, total, label):
    cur = a * b
    cur_inkb = cur / 1024
    total_inkb = total / 1024
    print("%s %dKB/%dKB  %.2f%%" % (label, cur_inkb, total_inkb, cur / float(total) * 100), end='\r')


try:
    with open('mnist_train.csv', 'r') as f:
        pass
except IOError:
    urllib.urlretrieve('http://www.pjreddie.com/media/files/mnist_train.csv', 'mnist_train.csv',
                       lambda a, b, c: print_process(a, b, c, 'Download Mnist Train CSV '))

try:
    with open('mnist_test.csv', 'r') as f:
        pass
except IOError:
    urllib.urlretrieve('http://www.pjreddie.com/media/files/mnist_test.csv', 'mnist_test.csv',
                       lambda a, b, c: print_process(a, b, c, 'Download Mnist Validation CSV'))

import matplotlib.pyplot as plt
import theanets

from utils import plot_layers


N = 16

e = theanets.Experiment(
    theanets.Classifier,
    layers=(784, N * N, 10),
    train_batches=100,
    train='train.conf',
    valid='valid.conf'
)
e.run()

plot_layers(e.network.weights)
plt.tight_layout()
plt.show()
