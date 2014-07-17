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
from .normalizers import VoidNormalizer
import csv
import abc
import numpy as np
import yaml

try:
    import cStringIO as StringIO
except ImportError:
    import StringIO

import linecache
import random
import multiprocessing


logging = climate.get_logger(__name__)


class IDataset(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __iter__(self):
        """
        The main method for dataset object. Return a 2D list-like object. First is input data matrix, second is ideal
        labels or ideal outputs.
        """
        pass


class SequenceDataset(IDataset):
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
        super(SequenceDataset, self).__init__()
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

    if reduce(lambda a, b: a and b, map(lambda x: isinstance(x, VoidNormalizer), normalizers)):
        normalizers = None

    return normalizers


class CSVDataset(IDataset):
    def __iter__(self):
        if self.inner_dataset is None:
            return self.__iter_impl__()
        else:
            return self.inner_dataset.__iter__()

    def __iter_impl__(self):
        if self.batches is None:
            # From First To Last.
            with open(self.filename, 'r') as f:
                io = StringIO.StringIO()
                for i, line in enumerate(f):
                    if (i + 1) % self.batch_size == 0:
                        io.flush()
                        yield self.__gen_data_from_lines__(io)
                        io.close()
                        io = StringIO.StringIO()
                    else:
                        print >> io, line
                io.close()
        else:
            for _ in range(self.batches):
                io = StringIO.StringIO()
                for __ in range(self.batch_size):
                    print >> io, linecache.getline(self.filename, random.randrange(0, self.total_line_count))
                yield self.__gen_data_from_lines__(io)

    def __gen_data_from_lines__(self, io):
        labels = []
        dim = [-1]

        def callback(d, lbl):
            dim.append(d - 1)
            for l in lbl:
                labels.append(l)

        def gen():
            io.seek(0)
            reader = csv.reader(io)
            return CSVDataset.__get_element_from_file__(reader, self.label_first, self.norms, callback)

        input_data = np.fromiter(gen(), dtype=np.float32)
        labels = np.asarray(labels, "int32")
        # print dim
        input_data = input_data.reshape(dim)
        return input_data, labels

    def __init__(self, label, exp, obj, norms):
        super(CSVDataset, self).__init__()

        self.batch_size = exp.args.batch_size
        load_in_mem = obj.get('loadIntoMem', False)

        self.inner_dataset = None
        if load_in_mem:
            # Load Dataset Into Memory & Normalize
            self.inner_dataset = CSVDataset.__load_csv_into_mem(label, exp, obj, norms)
        else:
            label_pos = obj.get('label', 'first')
            if label_pos == 'first':
                self.label_first = True
            else:
                self.label_first = False
            self.batches = getattr(exp.args, '%s_batches' % label, None)
            self.batch_size = exp.args.batch_size
            # print self.batch_size, self.batches
            self.filename = obj.get('file')
            with open(self.filename, 'r') as f:
                self.total_line_count = sum([1 for l in f])
            self.norms = norms


    @staticmethod
    def __get_element_from_file__(reader, label_first, norms, callback):
        d = None
        labels = []
        __gen__ = (x for x in reader if x.__len__() != 0)
        for line in __gen__:
            if d is None:
                d = len(line)
            if label_first:
                labels.append(line[0])
                line = line[1:]
            else:
                labels.append(line[-1])
                line = line[:-1]
            if norms is not None:
                for v in map(lambda x: np.float32(x[0].apply(float(x[1]))), zip(norms, line)):
                    yield v
            else:
                for v in line:
                    yield np.float32(v)
        callback(d, labels)


    @staticmethod
    def __load_csv_into_mem(label, exp, obj, norms):
        """
            Load CSV File --> Normalize --> SequenceDataset
        """
        filename = obj.get('file')
        # def csv_loader
        label_pos = obj.get('label', 'first')
        if label_pos == 'first':
            label_first = True
        else:
            label_first = False

        labels = []

        def get_element_from_csv():
            def callback(dim, lbl):
                CSVDataset.__load_csv_into_mem.dimension = dim - 1
                for l in lbl:
                    labels.append(l)

            with open(filename, 'r') as f:
                for i in CSVDataset.__get_element_from_file__(csv.reader(f), label_first, norms, callback):
                    yield i

        input_data = np.fromiter(get_element_from_csv(), dtype=np.float32)
        dimension = CSVDataset.__load_csv_into_mem.dimension
        input_data = input_data.reshape((-1, dimension))
        # print input_data[0]
        labels = np.asarray(labels, 'int32')
        kwargs = {}
        if 'batches' not in kwargs:
            b = getattr(exp.args, '%s_batches' % label, None)
            kwargs['batches'] = b
        if 'size' not in kwargs:
            kwargs['size'] = exp.args.batch_size
        kwargs['label'] = label
        return SequenceDataset(*(input_data, labels), **kwargs)


def load_dataset(label, exp, config_file):
    if config_file is not None:
        if isinstance(config_file, str):
            with open(config_file, 'r') as f:
                obj = yaml.load(f)
                normalizers = __load_normalizers(obj)
                tp = obj.get('type')
                if tp == 'csv':
                    ds = CSVDataset(label, exp, obj, normalizers)
                    exp.datasets[label] = ds