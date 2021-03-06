from abc import ABCMeta, abstractmethod

__author__ = 'reyoung'


class INormalizer:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def need_scan(self):
        pass

    @abstractmethod
    def scan(self, i, line):
        pass

    @abstractmethod
    def apply(self, x):
        pass


class VoidNormalizer(INormalizer):
    def apply(self, x):
        return x

    def need_scan(self):
        return False

    def scan(self, i, line):
        pass

    def __init__(self):
        super(VoidNormalizer, self).__init__()


class MinMaxNormalizer(INormalizer):
    def apply(self, x):
        # print x
        val = (self.output_max - self.output_min) / (self.input_max - self.input_min) * (
            x - self.input_max) + self.output_max
        return val

    def need_scan(self):
        return self.output_min is None or self.output_max is None

    def scan(self, i, line):
        x = float(line[i])
        if self.output_max is None or self.output_max < x:
            self.output_max = float(x)

        if self.output_min is None or self.output_min > x:
            self.output_min = float(x)

    def __init__(self, obj):
        super(MinMaxNormalizer, self).__init__()
        self.input_max = obj.get('input_max', None)
        if self.input_max is not None:
            self.input_max = float(self.input_max)

        self.input_min = obj.get('input_min', None)
        if self.input_min is not None:
            self.input_min = float(self.input_min)

        self.output_max = obj.get('output_max', None)
        if self.output_max is not None:
            self.output_max = float(self.output_max)

        self.output_min = obj.get('output_min', None)
        if self.output_min is not None:
            self.output_min = float(self.output_min)


def factory(obj):
    method = obj.get('method', 'void')
    if method == 'void':
        return VoidNormalizer()
    elif method == 'minmax':
        return MinMaxNormalizer(obj)
