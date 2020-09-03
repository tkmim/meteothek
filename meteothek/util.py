# -*- coding: utf-8 -*-
"""
"""

import numpy as np


def date_format(yyyy,mm,dd,hh):

    return str('%04d%02d%02d%02d%02d' % yyyy, mm, dd, hh)

def mkcmap(colors, levels):
    from matplotlib.colors import ListedColormap, BoundaryNorm

    #cm = LinearSegmentedColormap.from_list('custom_cmap', list_cid, N=len(list_cid))
    cm = ListedColormap(colors, name="custom")        
    norm = BoundaryNorm(levels, len(levels))

    return cm, norm

