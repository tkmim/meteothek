# -*- coding: utf-8 -*-
"""

"""

from . import util
import cartopy.crs as ccrs

def Regular_latlon():
    # somewhat this returned projection obejct does not work with 'transform' cartopy option. In that case, use ccrs.PlateCarree() directly in a plotting code.
    return ccrs.PlateCarree()

def Lambert_MSM_JP():
    return ccrs.LambertConformal(central_longitude=140, central_latitude=40, standard_parallels=(30,60))

def Lambert_D2_DE():
    return ccrs.LambertConformal(central_longitude=10, central_latitude=51, standard_parallels=(48,53))

def Rotated_DE():
    return ccrs.RotatedPole(pole_latitude=40, pole_longitude=-170)

