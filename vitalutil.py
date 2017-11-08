import logging
import re
import sys
import collections
from itertools import islice

logging.basicConfig(format='%(asctime)-15s : %(message)s',
                    level=logging.DEBUG)


def get_logger(name):
    return logging.getLogger(name)


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def create_square(array_shape, start, shape):
    """
    Gets the indices for a square while being safe
    :param array_shape: (H,W) from the array
    :param start: (x,y) center of the square
    :param shape: int the padding of the square
    :return: [(x,y)] list of indices.
    """
    indices = []
    x_s, y_s = start
    h, w = array_shape[0], array_shape[1]
    for i in range(int(x_s - shape), int(x_s + shape) + 1):
        for j in range(int(y_s - shape), int(y_s + shape) + 1):
            if i < 0 or j < 0 or i >= w or j >= h:
                continue
            indices.append((i, j))
    return indices


def conv_int(i):
    """
    Convert string into int if i is convertible.

    Parameters
    ----------
    i: str
	String to be converted into int.

    Returns
    -------
    int if possible, otherwise the string itself.
    """
    return int(i) if i.isdigit() else i


def natural_order(sord):
    """
    Split the sord string into a list of string and int.
    Use it as the key parameter of the sorted() function.

    Ex:

    ['1','10','2'] -> ['1','2','10']

    ['abc1def','ab10d','b2c','ab1d'] -> ['ab1d','ab10d', 'abc1def', 'b2c']

    Parameters
    ----------
    sord: str
    	String that needs to be sorted.

    Returns
    -------
    List of string and int.

    """
    if isinstance(sord, tuple):
        sord = sord[0]
    return [conv_int(c) for c in re.split(r'(\d+)', sord)]


def write_summary_to_file(model, filename):
    """Write the summary of a Keras model to a text file for viewing."""
    orig_stdout = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f
        model.summary()
        sys.stdout = orig_stdout


class OrderedSet(collections.MutableSet):
    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]  # sentinel node for doubly linked list
        self.map = {}  # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)
