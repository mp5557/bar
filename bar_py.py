# import doctest
from typing import Iterable, Iterator
from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ParaConverter(ABC):
    @abstractmethod
    def toPara(self, x: float) -> np.ndarray:
        pass

    @abstractproperty
    def dim(self) -> int:
        pass

    def eval(self, x: float, p: np.ndarray) -> Iterator[float]:
        return self.toPara(x) @ p

    def eval_range(self, x: Iterable[float], p: np.ndarray) -> Iterator[float]:
        return (self.toPara(xx) @ p for xx in x)

    def eval_np(self, x: Iterable[float], p: np.ndarray) -> Iterator[float]:
        return np.array(list(self.toPara(xx) @ p for xx in x))


class LineConverter(ParaConverter):
    '''
    >>> line_conv = LineConverter()
    >>> x = np.arange(3)
    >>> p = np.array([1, 3])
    >>> y = line_conv.eval_range(x, p)
    >>> print(list(y))
    [3, 4, 5]
    >>> yn = line_conv.eval_np(x, p)
    >>> print(yn)
    [3 4 5]
    '''

    def toPara(self, x: float) -> np.ndarray:
        return np.array([x, 1])

    @property
    def dim(self) -> int:
        return 2


class CurveFitter:
    def __init__(self, converter: ParaConverter):
        self.converter = converter
        dim = converter.dim
        self.ata = np.zeros((dim, dim))
        self.atb = np.zeros(dim)

    def update(self, x: float, y: float, w = 1.) -> None:
        vec = self.converter.toPara(x)
        self.ata += np.outer(vec, vec)
        self.atb += y * vec

    def solve(self) -> np.ndarray:
        return la.pinv(self.ata) @ self.atb


if __name__ == '__main__':
    # doctest.testmod()
    line_conv = LineConverter()
    line_conv.toPara(1)

    x = np.arange(10) / 5
    y = 2 * x + 1
    z = 3 * x + 1
    s = x
    df = pd.DataFrame({'x':x, 'y':y, 'z': z, 's':s})
    df.set_index('x')

    dfq = df[x < 1]

    sns.set()
    # scatter = lambda:sns.scatterplot(x=x, y=y, hue=s)
    def scatter():
        sns.scatterplot(x=x, y=y, hue=s)

    scatter()
    plt.show()

    y *= 2
    scatter()
    plt.show()

    # reviews = pd.read_csv("./winemag-data-130k-v2.csv")
