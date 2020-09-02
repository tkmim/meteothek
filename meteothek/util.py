# -*- coding: utf-8 -*-
"""
"""

import numpy as np


class util:
    
    def __init__(self):
        pass

    def date_format(yyyy,mm,dd,hh):
        time_list = []
        num_time_list = len(time_list)
        
        month_thirtyone = [ 1, 3, 5, 7, 8, 10, 12 ]
        month_thirty    = [ 4, 6, 9, 11 ]
        month_twntynine = [ 2 ]

        while num_time_list < 3:
            time_list.append(str(yyyy) + str('%02d' % mm) + str('%02d' % dd) + str('%02d' % hh) + '00')
            dd = dd - 1
    
            if dd < 0:
                mm = mm - 1
                if mm in month_thirty:
                    dd = 30
                elif mm in month_thirtyone:
                    dd = 31
                elif mm in month_twntynine:
                    if yyyy % 4 == 0:
                        dd = 28
                    else:
                        dd =29

            num_time_list += 1

        return time_list

    def mkcmap(colors, levels):
        from matplotlib.colors import ListedColormap, BoundaryNorm

        #cm = LinearSegmentedColormap.from_list('custom_cmap', list_cid, N=len(list_cid))
        cm = ListedColormap(colors, name="custom")        
        norm = BoundaryNorm(levels, len(levels))

        return cm, norm

