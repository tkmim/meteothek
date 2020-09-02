# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import glob
import f90nml



class read:
    
    def __init__(self):
        pass

    def nusdas(filepath):
        pass


    def simple_binary(filepath,nx,ny,nz,nt):
    # "simple binary" means a binary file comprised of one element with x, y, z, t dimenstions.
    #
        with open(filepath,'rb') as ifile:
            #data=np.fromfile(filepath,dtype='float32',sep='').reshape(nx,ny,nz,nt,order='F')
            data=np.fromfile(filepath,dtype='float32',sep='').reshape(nt,nz,nx,ny,order='F')
            data=data.transpose(2,3,1,0)

        return data


    def complex_binary(filepath,nx,ny,nz,nt):
    # "simple binary" means a binary file comprised of one element with x, y, z, t dimenstions.
    #
        with open(filepath,'rb') as ifile:
            data=np.fromfile(filepath,dtype='float32',sep='').reshape(nt,nz,nx,ny,order='F')
            data=data.transpose(2,3,1,0)

        return data


    def latlon_lambert(nx,ny):
        indir = '/home/da01/matsunobu/Toyotool/Nhm/test'
        grd_dir = '/Coordinate/Nhm_grd/'
        ifile_lat = indir + grd_dir + '400_lat_lambertgrd.txt'
        ifile_lon = indir + grd_dir + '400_lon_lambertgrd.txt'

        with open(ifile_lat, 'r') as ifile:
            lat = [s.strip() for s in ifile.readlines()][::-1]
        with open(ifile_lon, 'r') as ifile:
            lon = [s.strip() for s in ifile.readlines()]

        #print('..... Basepoint Check on already known num. ' + str(xlat) + ',  ' + str(xlon))
        #print('..... Basepoint Check on open num.          ' + str(lat[int(xj-1)]) +',  ' + str(lon[int(x    i-1)]))

        lon, lat = np.meshgrid(lon, lat)
        return lon, lat

    def gpv_data_coef(path, ft_grd):
        namelist_path = path + '/Nhm_Data/namelist/' + ft_grd + '_grid_namelist.cnf'
        params = f90nml.read(namelist_path)
        
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
        
        hgt, elem = 16, 5

        undef = -999.000 

        return nx, ny, hgt, elem, xi, xj, xlat, xlon, dx, dy, slat1, slon1, slat2, slon2, undef


