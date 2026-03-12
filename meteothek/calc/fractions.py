"""
a"""

import numpy as np
import xarray as xr
import logging
from meteothek.util import *
from scipy import signal

# set up the logger
logger = logging.getLogger(__name__)


def convolve2Dimage(field, kernel, max_missing=0.25, method="integral", verbose=False):
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
    if method == "fft":
        valid_conv = signal.fftconvolve(valid_field, valid_kernel, mode="same")
    elif method == "integral":
        valid_conv = integralImage(valid_field, valid_kernel)
    else:
        raise Exception('Invalid convolution method: select "fft" or "integral" ')

    # valid_field is always 0/1 integer-valued, so the true count inside every
    # window is an exact integer.
    valid_conv = np.round(valid_conv) / (valid_kernel.shape[0] * valid_kernel.shape[1])
    # If valid_conv is less than valid_threshold, that (convoluted) grid point will be discarded.
    # Calculate the valid threshold here
    valid_threshold = 1.0 - max_missing

    # -- Performe convolution
    # Set np.nan to a float to avoid nan in fftconvolve
    field = np.where(valid_field == 1, field, 0)

    # Make a convolution and masking the result by <result_mask>
    if method == "fft":
        result = signal.fftconvolve(field, kernel, mode="same")
    elif method == "integral":
        result = integralImage(field, kernel)
    else:
        raise Exception('Invalid convolution method: select "fft" or "integral" ')

    # Normalise by dividing with the fraction of valid points in a kernel
    # Ensure the result is computed using only valid grid points when the kernel window contains missing values.
    # Replace zero values with NaN to avoid divide-by-zero warning
    valid_conv = np.where(valid_conv == 0, np.nan, valid_conv)
    result = result / valid_conv

    # Mask with <valid_threshold>. Values are kept only when a window kernel contains enough valid grid points.
    mask = valid_conv >= valid_threshold
    result = np.where(mask, result, np.nan)

    return result


def integralImage(field, kernel, table=None):
    """perform convolution using integral image (summed-area table) method.
    This is an alternative to FFT convolution, and can be faster for small kernels.
    Implementation based on Faggian et al. (2015, doi: 10.54302/mausam.v66i3.555) and corrected by Necker et al. (2024, doi: 10.1002/qj.4824)


    Parameter
    ---------
    field : 2d nd-darray,
        a field be convoluted.
    kernel : 2d nd-array,
        convolution kernel.
    table : 2d nd-array, optional
        pre-computed integral table. If not provided, it will be computed internally.

    Returns
    -------
    result : 2d nd-array, convolution.
    """

    assert np.ndim(field) == 2, "<field> needs to be 2D."
    assert np.ndim(kernel) == 2, "<kernel> needs to be 2D."
    assert kernel.shape[0] <= field.shape[0], "<kernel> size needs to <= <field> size."
    assert kernel.shape[1] <= field.shape[1], "<kernel> size needs to <= <field> size."
    assert kernel.shape[0] == kernel.shape[1], "<kernel> needs to be square."

    n = kernel.shape[0]
    lw = n // 2   # left/top half-width (index of kernel centre from left edge)
    rw = n - lw   # right/bottom half-width (exclusive upper extent)
    if lw < 1:
        return field.astype(float)

    rows, cols = field.shape

    if table is None:  # compute integral table if not provided
        # Accumulate in float64 to minimise cancellation errors in the SAT.
        table = np.float64(field).cumsum(1).cumsum(0)

    # Pad the prefix-sum table with a zero row on top and a zero column on the left.
    # padded[i+1, j+1] == table[i, j], and padded[0, :] == padded[:, 0] == 0.
    # This lets us use exclusive-bound SAT arithmetic without special-casing:
    #   sum[r0..r1) x [c0..c1) = T[r1,c1] - T[r0,c1] - T[r1,c0] + T[r0,c0]
    padded = np.zeros((rows + 1, cols + 1), dtype=np.float64)
    padded[1:, 1:] = table

    # Build 1-D index vectors (length rows / cols) instead of full N×N grids.
    # np.ix_ turns them into a pair of broadcast-compatible arrays so that
    # padded[np.ix_(r1, c1)] gives an (rows×cols) outer-product lookup without
    # any ravel_multi_index / np.take / flattening overhead.
    ri = np.arange(rows, dtype=int)
    cj = np.arange(cols, dtype=int)

    # Exclusive lower/upper bounds in padded space.
    # fftconvolve 'same' centres the kernel at index lw (= n//2) from its left
    # edge, so the window for output pixel i spans [i-lw, i-lw+n) in field space,
    # which maps to [i-lw+1, i-lw+n+1) = [i-lw, i+rw) in padded space after the
    # zero-row/col shift.  Clipping to [0, rows/cols] handles boundaries.
    r0 = np.clip(ri - lw, 0, rows)
    r1 = np.clip(ri + rw, 0, rows)
    c0 = np.clip(cj - lw, 0, cols)
    c1 = np.clip(cj + rw, 0, cols)

    return (padded[np.ix_(r1, c1)] + padded[np.ix_(r0, c0)]
            - padded[np.ix_(r0, c1)] - padded[np.ix_(r1, c0)])


def fss(fcst, obs, threshold, window, thrsd_type, max_missing=0.25, method="integral"):
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
        fhat = convolve2Dimage((fcst > thresh_fx), kernel, max_missing=max_missing, method=method)
        ohat = convolve2Dimage((obs > thresh_obs), kernel, max_missing=max_missing, method=method)
    else:
        fhat = fcst > thresh_fx
        ohat = obs > thresh_obs

    # Calculate FSS numerator and denominator
    num = np.nanmean(np.power(np.float64(fhat) - np.float64(ohat), 2))
    denom = np.nanmean(np.power(np.float64(fhat), 2) + np.power(np.float64(ohat), 2))

    mask = (~np.isnan(fhat)) & (~np.isnan(ohat))
    logger.info(
        f"max_missing={max_missing}, {np.sum(1-mask)} grid points ({(1-np.nanmean(mask))* 100:.1f} %) are dropped."
    )

    # return numerator, denominator, and FSS value
    return num, denom, 1.0 - num / denom
