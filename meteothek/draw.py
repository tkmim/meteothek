# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import SCM6
from .util import *




class draw:
    
    def __init__(self):
        pass

    def set_projection(ax, blat, tlat, llon, rlon):
        map = Basemap( projection='lcc', resolution="i", lat_0=30.0, lon_0=140.0,lat_1=30, lat_2=60, fix_aspect=(1,1), llcrnrlat=blat, urcrnrlat=tlat, llcrnrlon=llon, urcrnrlon=rlon)

        #map = plt.axes(projection=ccrs.LambertConformal(central_longitude=140, central_latitude=40, standard_parrallels=(30,60)))
        return map

    def coastline(map):
        # drawing coastlines 
        map.drawcoastlines(color='black', linewidth=0.5)

        # Drawing grid lines every 5 degrees
        map.drawmeridians(np.arange(0, 360, 5),  labels=[False, False, False, True], fontsize='small', color='gray', linewidth=0.5)
        map.drawparallels(np.arange(-90, 90, 5), labels=[True, False, False, False], fontsize='small', color='gray', linewidth=0.5)


    def shade(map, x, y, data, *, cmap=SCM6.batlow,  extend='both', **kwargs):
        print('Keywords attached: ',*kwargs)
        levels = kwargs.get('levels', False) 
        colors = kwargs.get('colors', False)

        if 'colors' in locals():
            cmap, norm = util.mkcmap(colors, levels)

        if 'levels' in locals() and 'colors' in locals():
            shade = map.contourf(x, y, data, levels, colors=colors, extend=extend)

        elif 'levels' in locals():
            shade = map.contourf(x, y, data, levels, cmap=cmap, extend=extend)

        else:
            shade = map.contourf(x, y, data, cmap=cmap, extend=extend)

        return shade

    def mesh(map, x, y, data, *, cmap=SCM6.batlow, norm=-9999, **kwargs):
        levels = kwargs.get('levels', False) 
        colors = kwargs.get('colors', False)

        if 'colors' in locals():
            cmap, norm = util.mkcmap(colors, levels)
            mesh = map.pcolormesh(x, y, data, cmap=cmap, norm=norm)
        else:
            mesh = map.pcolormesh(x, y, data, cmap=cmap)

        return mesh


    def contour(map, x, y, data, *, levels=-9999, colors="black", linestyles='-',linewidths=0.5):

        if levels == -9999:
            levels=np.linspace(np.min(data), np.max(data), 6)

        contour = map.contour(x, y, data, colors=colors, linestyles=linestyles, linewidths=linewidths, levels=levels)
        contour.clabel(fmt='%1.1f', fontsize=8)

        return contour


    def vector(map, x, y, u, v, *, skip=10, scale=1.0e-4 ):
        arr_skip=(slice(None,None,skip), slice(None,None,skip))
        map.quiver(x[arr_skip],y[arr_skip],u[arr_skip],v[arr_skip],angles='xy',scale_units='xy',scale=scale)


    def barb(map,x,y,u,v):
        pass

    def point(map, x, y, *, marker="o", markersize=10):
        map.plot(x, y, marker=marker, markersize=markersize)

    def text(map, x, y, string, *, size=20, color="black"):
        plt.text(x, y, string, size=size, color=color)

    def title(map, txt, *, loc='left', fontsize=10):
        plt.title(txt, loc=loc, fontsize=fontsize)
  
    def colorbar(fig, cs, position):
            cax=fig.add_subplot(1,1,1)
            cax_pos = [position[0], position[2], 
                    position[1]-position[0], position[3]-position[2]]
            cax.set_position(cax_pos)
            fig.colorbar(cs, cax=cax)
