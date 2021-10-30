# -*- coding: utf-8 -*-
"""

"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import SCM6
from . import util, proj

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter,LongitudeFormatter


def coastline(ax, *, coast_lw=0.5, border_lw=0.3 ):
    
    # draw coastlines 
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=coast_lw)
    ax.add_feature(cfeature.BORDERS, linewidth=border_lw)

    
def gridlines(ax, *, label=True, latint=15, lonint=15, linewidth=1.0):


    # define gridline intervals
    xticks=np.arange(-180.0, 360.1, lonint)
    yticks=np.arange(-90,90.1, latint)
    #gl.xlocator = mticker.FixedLocator(xticks) 
    #gl.ylocator = mticker.FixedLocator(yticks)
    
    lonfmt = LongitudeFormatter(number_format='.1f',
                                           degree_symbol='',
                                           dateline_direction_label=True)
    latfmt = LatitudeFormatter(number_format='.1f',
                                          degree_symbol='')
    
    gl = ax.gridlines(draw_labels=label,
                xlocs=xticks, ylocs=yticks,
                xformatter=lonfmt, yformatter=latfmt,
                linewidth=linewidth,
                x_inline=False, y_inline=False)
    
    gl.top_labels = gl.right_labels = False
        

def shade(ax, lons, lats, data, *, cmap=SCM6.batlow, extend='both', **kwargs):

    print('draw.shade: keywords attached: ',*kwargs)
    kwargs.setdefault("projection", ax.projection)
    projection = kwargs.get('projection', False)
    
    # Unfold kwargs
    keywords = {}
    if 'scale' in kwargs and kwargs['scale'] == 'log':
        keywords['locator']=mticker.LogLocator()
    if 'colors' in kwargs:
        cmap = util.mkcmap(kwargs['colors'])
        if not 'levels' in kwargs or len(kwargs['levels']) != len(kwargs['colors']) + 1:
            print('Error: optional keyword "colors" must be used with "levels" and their lengths must be n and n+1')
            sys.exit()
            
    if 'levels' in kwargs:
        keywords['levels']=kwargs['levels']
        cmap = util.nlcmap(cmap, keywords['levels'])
        
    shade = ax.contourf(lons, lats, data, cmap=cmap, extend=extend, transform=projection, **keywords)

    return shade


def mesh(ax, lons, lats, data, *, cmap=SCM6.batlow, **kwargs):
    
    print('draw.mesh: keywords attached: ',*kwargs)
    kwargs.setdefault("projection", ax.projection)
    projection = kwargs.get('projection', False)
    
    # Unfold kwargs
    keywords = {}
    if 'scale' in kwargs and kwargs['scale'] == 'log':
        keywords['locator']=mticker.LogLocator()        
    if 'colors' in kwargs:
        cmap = util.mkcmap(kwargs['colors'])
        if not 'levels' in kwargs or len(kwargs['levels']) != len(kwargs['colors']) + 1:
            print('Error: optional keyword "colors" must be used with "levels" and their lengths must be n and n+1')
            sys.exit()    
    if 'levels' in kwargs:
        import matplotlib.colors
        keywords['norm'] = matplotlib.colors.BoundaryNorm(kwargs['levels'], cmap.N) 
        
    mesh = ax.pcolormesh(lons, lats, data, cmap=cmap, transform=projection, **keywords)

    return mesh


def contour(ax, lons, lats, data, *, colors="black", 
            labels=True, linestyles='-',linewidth=0.5, cfmt='%1.1f', **kwargs):
    
    print('draw.contour: keywords attached: ',*kwargs)
    kwargs.setdefault("projection", ax.projection)
    projection = kwargs.get('projection', False)  
    
    # Unfold kwargs  
    keywords = {}
    if 'levels' in kwargs:
        keywords['levels'] = kwargs['levels']
    else:
        keywords['levels'] = np.linspace(np.min(data), np.max(data), 6)
        
    contour = ax.contour(lons, lats, data, colors=colors, linestyles=linestyles, linewidths=linewidth,
                         transform=projection, **keywords)
    
    if labels == True:
        contour.clabel(fmt=cfmt, fontsize=8)

    return contour


def vector(ax, lons, lats, u, v, *, skip=10, scale=1.0, legend=True, U=5, labelunit='m/s', colors='black', **kwargs ):
    """
    
    param: scale: arrow size scale factor (e.g. scale=1.0, wind speed 10.0 m/s is depicted as 1 cm)
    """

    print('Keywords attached: ',*kwargs)
    kwargs.setdefault("projection", ax.projection)
    projection = kwargs.get('projection', False)   

    # Unfold kwargs
    
    
    scale=scale * 0.0393701
    # avoid zero division
    if scale == 0:
        print('Error: "scale" must not be zero')
        sys.exit() 
        
    arr_skip=(slice(None,None,skip), slice(None,None,skip))
    
    vector = ax.quiver(lons[arr_skip], lats[arr_skip], u[arr_skip], v[arr_skip],
                        angles='uv', scale_units='inches', scale=1/scale,
                        color=colors, zorder=5, transform=projection)
    
    if(legend):
        plt.gca().quiverkey(vector, 0.9, 1.05, U=U, label='%s %s' % (U, labelunit), 
                labelpos='E', coordinates='axes')

    return vector

def barb(ax,x,y,u,v):
    pass

def point(ax, lons, lats, *, marker="o", markersize=10, colors='blue'):
    """ Plot a point on a map """
    
    ax.plot(lons, lats, marker=marker, markersize=markersize, color=colors, 
            transform=ccrs.Geodetic()._as_mpl_transform(ax))

def text(ax, lons, lats, string, *, size=20, colors="black"):
    """ Draw text on a map """
    
    ax.text(lons, lats, string, size=size, color=colors,
            transform=ccrs .Geodetic()._as_mpl_transform(ax))

def title(ax, txt, *, loc='left', fontsize=10):
    plt.title(txt, loc=loc, fontsize=fontsize)

    
def colorbar(fig, cs, position, orientation='vertical', ticks=False):
    """
    param: position: [ left, right, bottom, top] #values between 0.0-1.0
    """
    keywords = {}
    if ticks is not False:
        keywords['ticks'] = ticks
         
    cax=fig.add_subplot(1,1,1, label='cax')
    cax_pos = [position[0], position[2], 
        position[1]-position[0], position[3]-position[2]]
    cax.set_position(cax_pos)
    fig.colorbar(cs, cax=cax, orientation=orientation, **keywords)

    
def add_subplot_axes(ax, rect, *, facecolor='w'):
    """
    param: rect[]
    """
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)

    return subax
