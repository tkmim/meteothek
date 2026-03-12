import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from meteothek.calc.fractions import convolve2Dimage
import logging

from meteothek.calc import fss

# showing DEBUG level messages
logging.basicConfig(level=logging.DEBUG)  # , format='%(levelname)s - %(name)s - %(asctime)s - %(message)s')

# showing messages in and above the INFO level in pynus.decode
logging.getLogger("meteothek.calc.fractions").setLevel(logging.DEBUG)

def make_field(shape=(10, 10), nan_fraction=0.1, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.rand(*shape)
    # introduce some NaNs
    mask = rng.rand(*shape) < nan_fraction
    a[mask] = np.nan
    return a

def test_convolve2dfft_basic():
    field = make_field((10, 10), nan_fraction=0.2, seed=1)
    kernel = np.ones((3, 3))
    result = convolve2Dimage(field, kernel, max_missing=0.5, method="fft")
    # shape preserved
    assert result.shape == field.shape
    # result is float and contains nans where appropriate
    assert np.issubdtype(result.dtype, np.floating)
    assert np.isnan(result).sum() >= 0

def test_fss_basic_accumulation():
    # create simple obs and forecast with a clear hotspot
    obs = np.zeros((10, 10))
    fcst = np.zeros((10, 10))
    obs[4:7, 4:7] = 1.0         # obs central square
    fcst[5:8, 5:8] = 1.0        # fcst shifted by one
    # include some NaNs at edges
    obs[0, :] = np.nan
    fcst[:, 0] = np.nan

    num, denom, score = fss(fcst, obs, threshold=0.5, window=3, thrsd_type="accumulation", max_missing=0.5)
    print(num, denom, score)
    # basic sanity checks
    assert num >= 0
    assert denom > 0
    assert 0.0 <= score <= 1.0

# Tolerance for floating-point comparison between integralConv and convolve2DFFT
_ABS_TOL = 1e-10

def test_integralConv_odd_windows():
    """integralConv must match convolve2DFFT for odd kernel sizes (3, 5, 7, 9)."""
    field = make_field((30, 30), nan_fraction=0.1, seed=10)
    for n in [3, 5, 7, 9]:
        kernel = np.ones((n, n))
        ref = convolve2Dimage(field, kernel, max_missing=0.5, method="fft")
        res = convolve2Dimage(field, kernel, max_missing=0.5, method="integral")
        diff = np.nanmax(np.abs(ref - res))
        assert diff < _ABS_TOL, (
            f"odd window n={n}: max abs diff {diff:.3e} exceeds tolerance {_ABS_TOL}"
        )

def test_integralConv_even_windows():
    """integralConv must match convolve2DFFT for even kernel sizes (2, 4, 6, 8)."""
    field = make_field((30, 30), nan_fraction=0.1, seed=20)
    for n in [2, 4, 6, 8]:
        kernel = np.ones((n, n))
        ref = convolve2Dimage(field, kernel, max_missing=0.5, method="fft")
        res = convolve2Dimage(field, kernel, max_missing=0.5, method="integral")
        diff = np.nanmax(np.abs(ref - res))
        assert diff < _ABS_TOL, (
            f"even window n={n}: max abs diff {diff:.3e} exceeds tolerance {_ABS_TOL}"
        )

def test_integralConv_matches_convolve2DFFT_with_nans():
    """integralConv must match convolve2DFFT for a field with a dense NaN region."""
    field = make_field((40, 40), nan_fraction=0.2, seed=99)
    # also add a contiguous NaN block near an edge
    field[:3, :] = np.nan
    for n in [3, 4, 5, 6]:
        kernel = np.ones((n, n))
        ref = convolve2Dimage(field, kernel, max_missing=0.3, method="fft")
        res = convolve2Dimage(field, kernel, max_missing=0.3, method="integral")
        diff = np.nanmax(np.abs(ref - res))
        assert diff < _ABS_TOL, (
            f"dense-NaN case n={n}: max abs diff {diff:.3e} exceeds tolerance {_ABS_TOL}"
        )

if __name__ == "__main__":
    # run tests directly for quick manual checks
    test_convolve2dfft_basic()
    print("convolve2DFFT test passed")
    test_fss_basic_accumulation()
    print("fss test passed")
    test_integralConv_odd_windows()
    print("integralConv odd-window test passed")
    test_integralConv_even_windows()
    print("integralConv even-window test passed")
    test_integralConv_matches_convolve2DFFT_with_nans()
    print("integralConv vs convolve2DFFT NaN test passed")