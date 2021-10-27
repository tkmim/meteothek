# -*- coding: utf-8 -*-
"""

"""

import numpy as np


def simple_binary(filepath, nx, ny, nz, nt):
    # "simple binary" means a Fortran direct access file comprised of one element with fixed x, y, z, t dimenstions.
    # Records must be ordered as the GrADS binary data order.
    """
    def simple_binary(filepath, nx, ny, nz, nt):
    
        return data[x, y, z, t]
    """
    with open(path,'rb') as ifile:
        
        data=np.fromfile(filepath,dtype='float32',sep='').reshape(nt,nz,nx,ny,order='F')
        data=data.transpose(2,3,1,0)

    return data


def complex_binary(filepath,nx,ny,nz,nt):
    # "complex binary" means a binary file comprised of multiple variables with the same x, y, z, t dimenstions.
    # not implemented yet
    with open(filepath,'rb') as ifile:
        data=np.fromfile(filepath,dtype='float32',sep='').reshape(nt,nz,nx,ny,order='F')
        data=data.transpose(2,3,1,0)

    return data

def read_namelist_NHM_lambert(path, ft_grd):
    import f90nml
    params = f90nml.read(path)

    """
    + namelist
    xi, xj                        # num of x_grid, y_grid
    xlat, xlon, dx, dy,           # center of grid, d-resolution
    slat1, slon1, slat2, slon2
    """

    nx     = params['GRID_INFO']['nx']
    ny     = params['GRID_INFO']['ny']
    xi     = params['GRID_INFO']['xi']
    xj     = params['GRID_INFO']['xj']
    xlat   = params['GRID_INFO']['xlat']
    xlon   = params['GRID_INFO']['xlon']
    dx     = params['GRID_INFO']['dx']
    dy     = params['GRID_INFO']['dy']
    slat1  = params['GRID_INFO']['slat1']
    slon1  = params['GRID_INFO']['slon1']
    slat2  = params['GRID_INFO']['slat2']
    slon2  = params['GRID_INFO']['slon2']

    undef = -999.000 

    return nx, ny, hgt, elem, xi, xj, xlat, xlon, dx, dy, slat1, slon1, slat2, slon2, undef


