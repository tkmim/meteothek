# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import time
from scipy import signal
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

class timer(object):
    """ Timer """
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
        if self.verbose and self.msecs > 1000 :
            print(self.name, ': elapsed time: %.3f s' % (self.msecs / 1000.0) )
        elif self.verbose :
            print(self.name, ': elapsed time: %.3f ms' % self.msecs)

def flush():
    """Flush standard outputs to an external file"""
    import sys
    sys.stdout.flush()

def date_to_ymdh(yyyy,mm,dd,hh):
    """Format date to YYYYMMDDHH"""
    return str('%04d%02d%02d%02d%02d' % yyyy, mm, dd, hh)


def mkcmap(colors):
    """Make a color map from listed colours"""
    cmap = ListedColormap(colors, name="custom")        

    return cmap


class nlcmap(LinearSegmentedColormap):
    """
    Make a nonlinear colormap from given Colormap
    
    Usage: nlcmap = util.nlcmap(cmap, levels)
    """
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

    
def fss(fcst, obs, threshold, window, thrsd_type):
    """
    Compute the fraction skill score using convolution.
    :param fcst: nd-array, forecast field.
    :param obs: nd-array, observation field.
    :param window: integer, window size.
    :param thrsd_type: character, "accumulation" or "percentile"
    
    :return: tuple of FSS numerator, denominator and score.
    """
    def fourier_filter(field, n):
        return signal.fftconvolve(field, np.ones((n, n)))

    # Obtain the thresholds
    if thrsd_type == 'accumulation':
        thresh_fx = threshold
        thresh_obs = threshold
    elif thrsd_type == 'percentile':  # ~ 'percentile'
        thresh_fx = np.nanpercentile(fcst, threshold)
        thresh_obs = np.nanpercentile(obs, threshold)
    else:
        exit("Invalid FSS threshold type: select \"accumulation\" or \"percentile\" ")

    fhat = fourier_filter(fcst > thresh_fx, window)
    ohat = fourier_filter(obs > thresh_obs, window)

    num = np.nanmean(np.power(fhat - ohat, 2))
    denom = np.nanmean(np.power(fhat, 2) + np.power(ohat, 2))

    return num, denom, 1. - num / denom
    
