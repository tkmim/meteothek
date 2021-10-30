# -*- coding: utf-8 -*-
"""
"""
import sys
import numpy as np
import time

from scipy import signal
from matplotlib.colors import ListedColormap, LinearSegmentedColormap



#:
#:   Logging and debugging 
#:

def flush():
    """Flush standard outputs"""
    sys.stdout.flush()
    
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




#:
#:   Colormap
#:

def mkcmap(colors, name="custom"):
    """Make a colormap from colours as a list object"""
    cmap = ListedColormap(colors, name=name)        

    return cmap


class nlcmap(LinearSegmentedColormap):
    """
    Make a nonlinear colormap from given Colormap
    
    *** Currently deprecated ***
    
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


#:
#:   Scientific computing
#:

def fss(fcst, obs, threshold, window, thrsd_type):
    """
    Compute the fraction skill score (Roberts and Lean, 2008) using convolution.
    :param fcst: nd-array, forecast field.
    :param obs: nd-array, observation field.
    :param window: integer, window size.
    :param thrsd_type: character, "accumulation" or "percentile"
    
    :return: tuple, (FSS numerator, denominator and score)
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
    

    
#:
#:   Miscellaneous
#:   

def show_stats(array, name="numpy.ndarray"):
    if type(array).__module__ != "numpy":
        sys.exit("Error: util.show_stat: not a numpy array")
    
    print('Stats of %s' % name)
    print('    ndim, shape', array.ndim, array.shape)
    print('    max, mean, min', np.max(array), np.mean(array), np.min(array))
    print('    nanmax, nanmean, nanmin', np.nanmax(array), np.nanmean(array), np.nanmin(array))

def date_to_ymdh(yyyy,mm,dd,hh):
    """Format date (year, month, day, hour) to YYYYMMDDHH"""
    return str('%04d%02d%02d%02d%02d' % yyyy, mm, dd, hh)