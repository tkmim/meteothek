"""

proj.py
=======

This module provides a set of functions that return cartopy projection objects.


"""

import cartopy.crs as ccrs


def Regular_latlon():
    # The standard latitude-longitude based projection having an equirectanglar cooredinate.

    # somewhat this returned projection obejct does not work with 'transform' cartopy option. In that case, use ccrs.PlateCarree() directly in a plotting code.
    return ccrs.PlateCarree()


def Lambert_MSM_JP():
    return ccrs.LambertConformal(central_longitude=140, central_latitude=40, standard_parallels=(30, 60))


def Lambert_D2_DE():
    return ccrs.LambertConformal(central_longitude=10, central_latitude=51, standard_parallels=(48, 53))


def Rotated_DE():
    # The rotated-pole projection suitable for vidualising ICON-D2 outputs.
    # The pole is located at 40N, 170W.
    return ccrs.RotatedPole(pole_latitude=40, pole_longitude=-170)


def PolarStereo_DE():
    return ccrs.Stereographic(true_scale_latitude=60.0, central_latitude=90.0, central_longitude=10.0)


def Geodetic():
    # This is not a map projection but rather a representation of the Earth's surface based on WGS84 (keeping the spherical shape).
    # The standard latitude-longitude system but accounting for spherical topology and geographical distance i.e. computationally expensive.
    return ccrs.Geodetic()
