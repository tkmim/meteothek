"""
a"""

import numpy as np
import xarray as xr
import logging
from meteothek.util import *
from scipy import signal

# set up the logger
logger = logging.getLogger(__name__)


def convolve2DFFT(field, kernel, max_missing=0.25, verbose=False):
    """perform safe 2D FFT convolution with NaN value handling.

    By nature of FFT, nan values are not allowed in fft convolution. This function performs convolution of a 2D field
    where nan values are replaced with zeros, and post processes the result 1) by applying a mask to drop off grid points
    where too many missing values are involved in the convolution window and 2) by normalising the result by the fraction of valid points in a kernel.

    Parameter
    ---------
    field : 2d nd-darray,
        a field be convoluted.
    kernel : 2d nd-array,
        convolution kernel.
    max_missing : float, default = 0.25
        max tolerable fraction of missings within a convolution window.
        e.g. if <max_missing> is 0.55, when over 55% of values within a given
        convolution window are missing, the center will be set as missing.
        If only 40% is missing, the center value will be computed using the
        remaining 60% data in the window.
        NOTE: out-of-bound grids are counted as missings, this is different
        from convolve2D(), where the number of valid values at edges drops
        as the kernel approaches the edge.
    verbose : boolean, optional, currently not used

    Returns
    -------
    result : 2d nd-array, convolution.
    """

    assert np.ndim(field) == 2, "<field> needs to be 2D."
    assert np.ndim(kernel) == 2, "<kernel> needs to be 2D."
    assert kernel.shape[0] <= field.shape[0], "<kernel> size needs to <= <field> size."
    assert kernel.shape[1] <= field.shape[1], "<kernel> size needs to <= <field> size."

    # --------------Generate the mask field--------------

    # Detect nan values in the given field (Valid =1, Nan=0)
    valid_field = 1 - np.isnan(field)
    # Detect nan values in the kernel (Valid =1, Nan=0) (usually kernel has no nan, but for security)
    valid_kernel = np.where(kernel == 0, 0, 1)
    # Calculate fractions of valid grid points in a kernel window
    # Note: mode='same' or 'full' does not affect FSS calculation, but 'same' is needed for the normalisation later.
    valid_conv = signal.fftconvolve(valid_field, valid_kernel, mode="same") / (
        valid_kernel.shape[0] * valid_kernel.shape[1]
    )
    # If valid_conv is less than valid_threshold, that (convoluted) grid point will be discarded.
    # Calculate the valid threshold here
    valid_threshold = 1.0 - max_missing

    # -- Performe convolution
    # Set np.nan to a float to avoid nan in fftconvolve
    field = np.where(valid_field == 1, field, 0)

    # Make a convolution and masking the result by <result_mask>
    result = signal.fftconvolve(field, kernel, mode="same")
    # Normalise by dividing with the fraction of valid points in a kernel
    # Ensure the result is computed using only valid grid points when the kernel window contains missing values.
    result = result / valid_conv

    # Mask with <valid_threshold>. Values are kept only when a window kernel contains enough valid grid points.
    mask = valid_conv >= valid_threshold
    result = np.where(mask, result, np.nan)

    return result


def fss(fcst, obs, threshold, window, thrsd_type, max_missing=0.25):
    """
    Compute fractions skill score (FSS; Roberts and Lean, 2008) using convolution.
    Implementation based on Faggian et al. (2015, doi: 10.54302/mausam.v66i3.555)


    Parameters
    ----------
    fcst : 2d nd-array,
        forecast field
    obs : 2d nd-array,
        observation field
    window : integer,
        neighbourhood window size
    thrsd_type : character,
        "accumulation" or "percentile"
    max_missing : float, optional
        max tolerable fraction of missings within a convolution window.
        The default is 0.25.

    Returns
    -------
    FSS : tuple, (FSS numerator, denominator, score)

    """

    assert np.ndim(fcst) == 2, "<fcst> needs to be 2D."
    assert np.ndim(obs) == 2, "<obs> needs to be 2D."
    assert fcst.shape == obs.shape, "<fcst> size must be equal to <obs> size."

    # Obtain the thresholds
    if thrsd_type == "accumulation":
        thresh_fx = threshold
        thresh_obs = threshold
    elif thrsd_type == "percentile":
        thresh_fx = np.nanpercentile(fcst, threshold)
        thresh_obs = np.nanpercentile(obs, threshold)
    else:
        # if thrsd_type is neither 'accumulation' nor 'percentile'
        raise Exception('Invalid FSS threshold type: select "accumulation" or "percentile" ')

    # Make a nan mask on the original grid
    fmask = np.where(~np.isnan(fcst), 1, np.nan)
    omask = np.where(~np.isnan(obs), 1, np.nan)

    kernel = np.ones((window, window))

    if window > 1:
        fhat = convolve2DFFT((fcst > thresh_fx), kernel, max_missing=max_missing)
        ohat = convolve2DFFT((obs > thresh_obs), kernel, max_missing=max_missing)
    else:
        fhat = fcst > thresh_fx
        ohat = obs > thresh_obs

    # Calculate FSS numerator and denominator
    num = np.nanmean(np.power(fhat - ohat, 2))
    denom = np.nanmean(np.power(fhat, 2) + np.power(ohat, 2))
    
    mask = (~np.isnan(fhat)) & (~np.isnan(ohat))
    logger.info(f"max_missing={max_missing}, {np.sum(1-mask)} grid points ({(1-np.nanmean(mask))* 100:.1f} %) are dropped.")

    # return numerator, denominator, and FSS value
    return num, denom, 1.0 - num / denom
