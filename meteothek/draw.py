# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import SCM6
from . import util


def set_projection(ax, blat, tlat, llon, rlon):
    map = Basemap( projection='lcc', resolution="i", lat_0=30.0, lon_0=140.0,lat_1=30, lat_2=60, fix_aspect=(1,1), llcrnrlat=blat, urcrnrlat=tlat, llcrnrlon=llon, urcrnrlon=rlon)

    #map = plt.axes(projection=ccrs.LambertConformal(central_longitude=140, central_latitude=40, standard_parrallels=(30,60)))
    return map

def coastline(map):
    # drawing coastlines 
    map.drawcoastlines(color='black', linewidth=0.5)
    map.drawcountries(linewidth=0.3, linestyle='solid', color='k', antialiased=1, ax=None, zorder=None)

    # Drawing grid lines every 5 degrees
    map.drawmeridians(np.arange(0, 360, 1),  labels=[False, False, False, True], fontsize='small', color='gray', linewidth=0.5)
    map.drawparallels(np.arange(-90, 90, 1), labels=[True, False, False, False], fontsize='small', color='gray', linewidth=0.5)


def shade(map, x, y, data, *, cmap=SCM6.batlow,  extend='both', **kwargs):
    print('Keywords attached: ',*kwargs)
    levels = kwargs.get('levels', False) 
    colors = kwargs.get('colors', False)

    if levels != False and colors != False:
        cmap, norm = util.mkcmap(colors, levels)

    if levels != False and colors != False:
        shade = map.contourf(x, y, data, levels, colors=colors, extend=extend)

    elif levels != False:
        shade = map.contourf(x, y, data, levels, cmap=cmap, extend=extend)

    else:
        shade = map.contourf(x, y, data, cmap=cmap, extend=extend)

    return shade

def mesh(map, x, y, data, *, cmap=SCM6.batlow, norm=-9999, **kwargs):
    levels = kwargs.get('levels', False) 
    colors = kwargs.get('colors', False)

    if colors != False:
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


def vector(map, x, y, u, v, *, skip=10, scale=1.0, legend=True ):
    scale=scale * 0.0393701
    # avoid zero division
    if scale == 0:
        scale = 0.0393701

    arr_skip=(slice(None,None,skip), slice(None,None,skip))
    vector = map.quiver(x[arr_skip],y[arr_skip],u[arr_skip],v[arr_skip],angles='xy',scale_units='inches',scale=1/scale)
    if(legend):
        plt.gca().quiverkey(vector, 0.9, 1.05, U=5, label='5 m/s', 
                labelpos='E', coordinates='axes')
    return vector

def barb(map,x,y,u,v):
    pass

def point(map, x, y, *, marker="o", markersize=10):
    map.plot(x, y, marker=marker, markersize=markersize)

def text(map, x, y, string, *, size=20, color="black"):
    plt.text(x, y, string, size=size, color=color)

def title(map, txt, *, loc='left', fontsize=10):
    plt.title(txt, loc=loc, fontsize=fontsize)

def colorbar(fig, cs, position):
        cax=fig.add_subplot(1,1,1, label='cax')
        cax_pos = [position[0], position[2], 
                position[1]-position[0], position[3]-position[2]]
        cax.set_position(cax_pos)
        fig.colorbar(cs, cax=cax)

