# -*- coding: utf-8 -*-
"""
"""
import sys
import numpy as np
import time
from scipy import signal
from scipy.fft import fft, ifft, dct, idct
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
        if self.verbose and self.msecs > 1000 * 60 :
            print(self.name, ': elapsed time: %.3f min' % (self.msecs / 60000.0) )
        elif self.verbose and self.msecs > 1000 :
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

def fourier_filter(field, n, max_missing=0.25):
    """A wrapper function for making 2D Fourier filter convolution

    Parameter
    ---------
    field : 2d nd-darray, 
        a field be convoluted.
    n : integer,
        a size of the kernel for FFT convolution

    Returns
    -------
    result : 2d nd-array, convolution
    """

    def convolve2DFFT(field, kernel, max_missing=0.25, verbose=False):
        """2D FFT convolution with NaN

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

        assert np.ndim(field)==2, "<field> needs to be 2D."
        assert np.ndim(kernel)==2, "<kernel> needs to be 2D."
        assert kernel.shape[0]<=field.shape[0], "<kernel> size needs to <= <field> size."
        assert kernel.shape[1]<=field.shape[1], "<kernel> size needs to <= <field> size."

        #--------------Get mask for missings--------------
        # Make an NaN/Valid field (Valid =1, Nan=0)
        fieldcount=1-np.isnan(field) 
        # Binarisation of the kernel (usually do nothing)
        kernelcount=np.where(kernel==0, 0, 1) 
        # Make a convolution of 'Valid' grid points 
        # Note: mode='same' or 'full' does not affect FSS calculation
        result_mask=signal.fftconvolve(fieldcount,kernelcount, mode='same') 

        # Calculate the number of grid points working as the threshold for valid convoluted grid points 
        valid_threshold=(1.-max_missing)*np.sum(kernelcount) 

        # Set np.nan to a float to avoid getting nan in the convlutions
        field=np.where(fieldcount==1,field,0)

        # Make a convolution and masking the result by <result_mask>
        result=signal.fftconvolve(field,kernel, mode='same')
        result=np.where(result_mask>=valid_threshold, result, np.nan) #

        return result

    return convolve2DFFT(field, np.ones((n, n)), max_missing)
    
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

    assert np.ndim(fcst)==2, "<fcst> needs to be 2D."
    assert np.ndim(obs)==2, "<obs> needs to be 2D."
    assert fcst.shape == obs.shape, "<fcst> size must be equal to <obs> size."

    # Obtain the thresholds
    if thrsd_type == 'accumulation':
        thresh_fx = threshold
        thresh_obs = threshold
    elif thrsd_type == 'percentile':  # ~ 'percentile'
        thresh_fx = np.nanpercentile(fcst, threshold)
        thresh_obs = np.nanpercentile(obs, threshold)
    else:
        # if thrsd_type is neither 'accumulation' nor 'percentile'
        raise Exception("Invalid FSS threshold type: select \"accumulation\" or \"percentile\" ")

    # Make a nan mask
    fmask = np.where(~np.isnan(fcst), 1, np.nan)
    omask = np.where(~np.isnan(obs), 1, np.nan)
    
    if window > 1:
        fhat = fourier_filter((fcst > thresh_fx)*fmask, window)
        ohat = fourier_filter((obs > thresh_obs)*omask, window)
    else:
        fhat = (fcst > thresh_fx)*fmask
        ohat = (obs > thresh_obs)*omask
        
    num = np.nanmean(np.power(fhat - ohat, 2))
    denom = np.nanmean(np.power(fhat, 2) + np.power(ohat, 2))

    # return numerator, denominator, and FSS value
    return num, denom, 1. - num / denom


def AS(fcst, obs, Slim=np.nan, alpha=0.5):
    """
    Compute the agreement scales (AS; Dey et al. 2016) using convolution.
    
    Parameters
    ----------
    fcst : 2d nd-array, 
        forecast field. Size must be identical to <obs>.
    obs : 2d nd-array, 
        observation field. Size must be identical to <fcst>.
    Slim : integer, 
        Maximum scale (the number of grid points), of which scale of smoothing
        always two compared grid points are well agreed.
    alpha : float, optional
        max tolerable fraction of missings within a convolution window.
        The default is 0.5. 
    
    Returns
    -------
    Agreement Scales : 2d nd-array, integer, 
        Agreement scales
    """

    assert np.ndim(fcst)==2, "<fcst> needs to be 2D."
    assert np.ndim(obs)==2, "<obs> needs to be 2D."
    assert fcst.shape == obs.shape, "<fcst> size must be equal to <obs> size."
    assert np.ndim(Slim) < 2, "<Slim> needs to be a scalar."

    Agreement_Scales = np.ones(fcst.shape) * np.nan

    # If S_lim is not given, S_lim is set to the maximum possible value fits the given 2D field
    if np.isnan(Slim):
        Slim = np.floor((np.max(fcst.shape) - 1) * 0.5).astype(np.int32)
        

    # iterate window size
    for window in range(Slim, -1, -1):
        kernel_size =  2 * window + 1
        if window > 0:
            fhat = fourier_filter(fcst, kernel_size)
            ohat = fourier_filter(obs, kernel_size)
        else:
            fhat = fcst
            ohat = obs

        # take a difference of two 'convoluted' fields
        Ds = (fhat - ohat) ** 2 / (fhat ** 2 + ohat ** 2)
        # If the denominator is exclusively small, DS is set to 0, means well-agreement.
        Ds = np.where((fhat ** 2 + ohat ** 2) < 0.0001, 0, Ds)

        # Is Ds small enough?
        Ds_crit = alpha + (1 - alpha) * window / Slim
        Agreement_Scales = np.where(Ds <= Ds_crit, window, Agreement_Scales)

    Agreement_Scales = Agreement_Scales * ~np.isnan(fcst) * ~np.isnan(obs)

    return Agreement_Scales



def calc_ke_dct(u, v, trunc=None, type=2):
    """
    ******** No longer maintained ********

    Compute spectral decomposition of kinetic energy using DCT (Denis et al. 2002, MWR)
    
    :param u: 2d-array, zonal wind field
    :param v: 2d-array, meridional wind field
    :param trunc: optional, integer, wave number for rhomboidal trunctation
    :param type:  optional, integer, computational method of DCT (default: 2)
    
    :return: kk[trunc, trunc]: 2d-array, rounded total wavenumber
    :return: Ek[trunc, trunc]: 2d-array, kinetic energy
    """
    uk = dct(dct(u, axis=0, type=type, norm='ortho'), axis=1, type=type, norm='ortho')
    vk = dct(dct(v, axis=0, type=type, norm='ortho'), axis=1, type=type, norm='ortho')
    
    Nm = uk.shape[0]
    Nn = uk.shape[1]
    
    kk = np.rint(np.sqrt( np.power(np.arange(Nm).reshape(Nm, 1),2) + np.power(np.arange(Nn).reshape(1, Nn),2)))
    Ek = (uk * uk + vk * vk ) * 0.5 
    
    return kk[:trunc, :trunc], Ek[:trunc, :trunc]
    
def calc_te_dct(u, v, trunc=None, type=2):
    """
    ******** No longer maintained ********
    
    Compute spectral decomposition of total energy using DCT (Denis et al. 2002, MWR)
    
    :param u: 2d-array, zonal wind field
    :param v: 2d-array, meridional wind field
    :param t: 2d-array, temperature field
    :param trunc: optional, integer, wave number for rhomboidal trunctation
    :param type:  optional, integer, computational method of DCT (default: 2)
    
    :return: kk[N]: 1d-array, total wavenumber
    :return: Ek[4,N]: 1d-array, [total energy, zonal wind energy, meridional wind energy, temperature term]
    """
    uk = dct(dct(u, axis=0, type=type, norm='ortho'), axis=1, type=type, norm='ortho')
    vk = dct(dct(v, axis=0, type=type, norm='ortho'), axis=1, type=type, norm='ortho')
    tk = dct(dct(dt, axis=0, type=type, norm='ortho'), axis=1, type=type, norm='ortho')
    
    # truncation
    uk = uk[:trunc, :trunc]
    vk = vk[:trunc, :trunc]
    tk = tk[:trunc, :trunc]
    
    Nm = uk.shape[0]
    Nn = uk.shape[1]
    
    kk = np.rint(np.sqrt( np.power(np.arange(Nm).reshape(Nm, 1),2) + np.power(np.arange(Nn).reshape(1, Nn),2)))
    kes = np.sort(np.unique(kk))
    
    Ek = np.zeros((4, kes.shape[0]))

    Ek[0,:] = np.array([np.nansum(np.where(kk == ii, (uk * uk + vk * vk + 1004.0 / 287.0 * tk * tk ) * 0.5, np.nan)) for ii in kes])
    Ek[1,:] = np.array([np.nansum(np.where(kk == ii, (uk * uk) * 0.5, np.nan)) for ii in kes])
    Ek[2,:] = np.array([np.nansum(np.where(kk == ii, (vk * vk) * 0.5, np.nan)) for ii in kes])
    Ek[3,:] = np.array([np.nansum(np.where(kk == ii, (1004.0 / 287.0 * tk * tk ) * 0.5, np.nan)) for ii in kes])

    
    return kes, Ek 
    
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
