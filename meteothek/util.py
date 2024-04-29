"""
This module contains utility functions for meteothek.
Classes and functions in this module can be directly called from the top-level of the package.
"""

import sys
import logging
import numpy as np
import time
import pandas as pd
from typing import Union

# set up the logger and set the logging level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#:
#:   Logging and debugging
#:
def Flush():
    """Flush standard outputs"""
    sys.stdout.flush()


class Timer(object):
    """A function to measure the time of a block of code

    Parameters
    ----------
    name : str
        Name of the block to measure.
    verbose : bool, optional
        If True, print the elapsed time. If False, log the elapsed time. Default is True.

    ===== Example =====

    with Timer('name of the block'):
        # code to measure

    """

    def __init__(self, name: str, verbose=True):
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
            if self.msecs > 1000 * 60:
                print(self.name, ": elapsed time: %.3f min" % (self.msecs / 60000.0))
            elif self.msecs > 1000:
                print(self.name, ": elapsed time: %.3f s" % (self.msecs / 1000.0))
            else:
                print(self.name, ": elapsed time: %.3f ms" % self.msecs)
        else:
            # if verbose is False, log the elapsed time
            if self.msecs > 1000 * 60:
                logger.info(self.name + ": elapsed time: %.3f min" % self.msecs / 60000.0)
            elif self.msecs > 1000:
                logger.info(self.name + ": elapsed time: %.3f s" % self.msecs / 1000.0)
            else:
                logger.info(self.name + ": elapsed time: %.3f ms" % self.msecs)


#:
#:   Miscellaneous
#:


def ymdh2date(ymdh: Union[str, list]):
    """
    Format date string YYYYMMDDHH to datetime object.

    Parameters
    ----------
    ymdh : str or list of str
        Date string(s) in the format 'YYYYMMDDHH'.

    Returns
    -------
    date : pandas.DatetimeIndex or list of pandas.DatetimeIndex
        Datetime object(s) corresponding to the input date string(s).

    """

    if isinstance(ymdh, str):
        date = pd.to_datetime(ymdh, format="%Y%m%d%H")
    elif isinstance(ymdh, list):
        # if ymdh is a list, convert each element to datetime object
        date = [pd.to_datetime(str(dd), format="%Y%m%d%H") for dd in ymdh]
    else:
        raise TypeError("The input date must be a string or a list of strings.")

    return date
