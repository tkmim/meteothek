import xarray as xr
from .xrboots import *


xr.DataArray.boot1d = boot1d
xr.DataArray.bootnd = bootnd
xr.Dataset.boot1d = boot1d
xr.Dataset.bootnd = bootnd


# the code to do bootstrapping resampling along multiple xarray axis.
# This is __init__.py for a package

