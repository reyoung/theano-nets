# Copyright (c) 2012 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''This file contains a class for handling batched datasets.'''

import climate
import collections
import numpy.random as rng
from .normalizers import factory as normalizer_factory
import csv
import yaml


logging = climate.get_logger(__name__)


class SequenceDataset(object):
    '''This class handles batching and shuffling a dataset.

    It's mostly copied from the dataset class from hf.py, except that the
    constructor has slightly different semantics.
    '''

    def __init__(self, *data, **kwargs):
        '''Create a minibatch dataset from a number of different data arrays.

        Positional arguments:

        There should be one unnamed keyword argument for each input in the
        neural network that will be processing this dataset. For instance, if
        you are dealing with a classifier network, you'll need one argument for
        the inputs (e.g., mnist digit pixels), and another argument for the
        target outputs (e.g., digit class labels). The order of the arguments
        should be the same as the order of inputs in the network. All arguments
        are expected to have the same number of elements along the first axis.

        Alternatively, if there is only one positional arg, and it is callable,
        then that callable will be invoked repeatedly at training and test time.
        Each invocation of the callable should return a tuple containing one
        minibatch of data. The callable will not be passed any arguments.

        Keyword arguments:

        size or batch_size: The size of the mini-batches to create from the
          data sequences. Defaults to 32.
        batches: The number of batches to yield for each call to iterate().
          Defaults to the length of the data divided by batch_size.
        label: A string that is used to describe this dataset. Usually something
          like 'test' or 'train'.
        '''
        self.label = kwargs.get('label', 'dataset')
        self.number_batches = kwargs.get('batches')
        self.batch = 0

        size = kwargs.get('size', kwargs.get('batch_size', 32))
        batch = None
        cardinality = None
        self.callable = None
        self.batches = None
        if len(data) == 1 and isinstance(data[0], collections.Callable):
            self.callable = data[0]
            cardinality = '->'
            batch = self.callable()
            if not self.number_batches:
                self.number_batches = size
        else:
            self.batches = [
                [d[i:i + size] for d in data]
                for i in range(0, len(data[0]), size)]
            self.shuffle()
            cardinality = len(self.batches)
            batch = self.batches[0]
            if not self.number_batches:
                self.number_batches = cardinality

        logging.info('data %s: %s mini-batches of %s',
                     self.label, cardinality, ', '.join(str(x.shape) for x in batch))

    def __iter__(self):
        return self.iterate(True)

    def shuffle(self):
        rng.shuffle(self.batches)

    def iterate(self, update=True):
        if self.callable:
            return self._iter_callable()
        return self._iter_batches(update)

    def _iter_batches(self, update=True):
        k = len(self.batches)
        for b in range(self.number_batches):
            yield self.batches[(self.batch + b) % k]
        if update:
            self.update()

    def _iter_callable(self):
        for b in range(self.number_batches):
            yield self.callable()

    def update(self):
        if self.callable:
            return
        self.batch += self.number_batches
        if self.batch >= len(self.batches):
            self.shuffle()
            self.batch = 0


def __load_normalizers(obj):
    base_normalizer = obj.get('base_normalizer', {
        'method': 'void'
    })

    # Try Load Data
    filename = obj['file']
    normalizers = None
    normalizers_config = obj.get('normalizers', {})

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        need_scan = None
        for line in reader:
            if normalizers is None:
                normalizers = [normalizer_factory(normalizers_config.get(int(i), base_normalizer)) for i in
                               range(len(line))]
            if need_scan is None:
                need_scan = reduce(lambda a, b: a and b, [n.need_scan() for n in normalizers])
                if not need_scan:
                    break
            # Scan
            for i, n in enumerate(normalizers):
                n.scan(i, line)
    return normalizers


class CSVDataset(object):
    def __init__(self, obj, norms):
        pass

def load_dataset(exp, config_file):
    if config_file is not None:
        if isinstance(config_file, str):
            with open(config_file, 'r') as f:
                obj = yaml.load(f)
                normalizers = __load_normalizers(obj)
                label = obj.get('label', config_file)
                tp = obj.get('type')
                if tp == 'csv':
                    ds = CSVDataset(obj, normalizers)