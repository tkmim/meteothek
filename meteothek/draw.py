# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import SCM6
from . import util, proj

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter,LongitudeFormatter

def coastline(map, *, coast_lw=0.5, border_lw=0.3 ):
    
    # draw coastlines 
    map.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=coast_lw)
    map.add_feature(cfeature.BORDERS, linewidth=border_lw)

    
def gridlines(map, *, label=True, latint=15, lonint=15):

    gl = map.gridlines(crs=map.projection, draw_labels=False, 
                  x_inline=False, 
                  y_inline=False)

    # define gridline intervals
    xticks=np.arange(-180.0, 360.1, lonint)
    yticks=np.arange(-90,90.1, latint)
    gl.xlocator = mticker.FixedLocator(xticks) 
    gl.ylocator = mticker.FixedLocator(yticks)

    if label == True:
        
        # draw labels
        map.set_xticks(xticks,crs=map.projection)
        map.set_yticks(yticks,crs=map.projection)


        lonfmt = LongitudeFormatter(number_format='.1f',
                                           degree_symbol='',
                                           dateline_direction_label=True)
        latfmt = LatitudeFormatter(number_format='.1f',
                                          degree_symbol='')

        map.xaxis.set_major_formatter(lonfmt)
        map.yaxis.set_major_formatter(latfmt)


def shade(map, lons, lats, data, *, cmap=SCM6.batlow, extend='both', projection=proj.Regular_latlon(), **kwargs):
    
    print('Keywords attached: ',*kwargs)
    levels = kwargs.get('levels', False) 
    colors = kwargs.get('colors', False)

    if levels != False and colors != False:
        cmap, norm = util.mkcmap(colors, levels)

    if levels != False and colors != False:
        shade = map.contourf(lons, lats, data, levels, colors=colors, extend=extend, transform=projection)

    elif levels != False:
        shade = map.contourf(lons, lats, data, levels, cmap=cmap, extend=extend, transform=projection)

    else:
        shade = map.contourf(lons, lats, data, transform=projection, cmap=cmap, extend=extend)

    return shade


def mesh(map, lons, lats, data, *, cmap=SCM6.batlow, norm=-9999, **kwargs):
    levels = kwargs.get('levels', False) 
    colors = kwargs.get('colors', False)

    if colors != False:
        cmap, norm = util.mkcmap(colors, levels)
        mesh = map.pcolormesh(lons, lats, data, cmap=cmap, norm=norm)
    else:
        mesh = map.pcolormesh(lons, lats, data, cmap=cmap)

    return mesh


def contour(map, lons, lats, data, *, levels=-9999, colors="black", linestyles='-',linewidths=0.5):

    if levels == -9999:
        levels=np.linspace(np.min(data), np.max(data), 6)

    contour = map.contour(lons, lats, data, colors=colors, linestyles=linestyles, linewidths=linewidths, levels=levels)
    contour.clabel(fmt='%1.1f', fontsize=8)

    return contour


def vector(map, lons, lats, u, v, *, skip=10, scale=1.0, legend=True ):
    scale=scale * 0.0393701
    # avoid zero division
    if scale == 0:
        scale = 0.0393701

    arr_skip=(slice(None,None,skip), slice(None,None,skip))
    vector = map.quiver(lons[arr_skip], lats[arr_skip], u[arr_skip], v[arr_skip],
                        angles='xy', scale_units='inches', scale=1/scale)
    
    if(legend):
        plt.gca().quiverkey(vector, 0.9, 1.05, U=5, label='5 m/s', 
                labelpos='E', coordinates='axes')
    return vector

def barb(map,x,y,u,v):
    pass

def point(map, lons, lats, *, marker="o", markersize=10, color='blue'):
    map.plot(lons, lats, marker=marker, markersize=markersize, color=color)

def text(map, lons, lats, string, *, size=20, color="black"):
    
    map.text(lons, lats, string, size=size, color=color,
            transform=ccrs.Geodetic()._as_mpl_transform(map))

def title(map, txt, *, loc='left', fontsize=10):
    plt.title(txt, loc=loc, fontsize=fontsize)

def colorbar(fig, cs, position):
        cax=fig.add_subplot(1,1,1, label='cax')
        cax_pos = [position[0], position[2], 
                position[1]-position[0], position[3]-position[2]]
        cax.set_position(cax_pos)
        fig.colorbar(cs, cax=cax)

