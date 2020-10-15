# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import time
from matplotlib.colors import LinearSegmentedColormap


class timer(object):
    def __init__(self, name, verbose=True):
        self.verbose = verbose
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print(self.name, ': elapsed time: %f ms' % self.msecs)


def date_format(yyyy,mm,dd,hh):
    """Format date to YYYYMMDDHH"""
    return str('%04d%02d%02d%02d%02d' % yyyy, mm, dd, hh)


def mkcmap(colors, levels):
    """Make a color map from listed colours"""
    from matplotlib.colors import ListedColormap, BoundaryNorm

    #cm = LinearSegmentedColormap.from_list('custom_cmap', list_cid, N=len(list_cid))
    cm = ListedColormap(colors, name="custom")        
    norm = BoundaryNorm(levels, len(levels))

    return cm, norm


class nlcmap(LinearSegmentedColormap):
    """A nonlinear colormap from given Colormap"""
    name = 'nlcmap'

    def __init__(self, cmap, levels):
        self.cmap = cmap
        self.monochrome = self.cmap.monochrome
        self.levels = np.asarray(levels, dtype='float64')
        self._x = self.levels-self.levels.min()
        self._x/= self._x.max()
        self._y = np.linspace(0, 1, len(self.levels))

    def __call__(self, xi, alpha=1.0, **kw):
        yi = np.interp(xi, self._x, self._y)
        return self.cmap(yi, alpha)
