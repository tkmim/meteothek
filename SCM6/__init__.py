"""
    ScientificColourMaps6

    Usage
    -----
    import ScientificColourMaps6 as SCM6
    plt.imshow(data, cmap=SCM6.berlin)

    Available colourmaps
    ---------------------
    acton, bamako, batlow, berlin, bilbao, broc, buda, cork, davos, devon,
    grayC, hawaii, imola, lajolla, lapaz, lisbon, nuuk, oleron, oslo, roma,
    tofino, tokyo, turku, vik


    License
    -------
    The Scientific Colour Maps are licensed under a MIT License
    Copyright (c) 2020, Fabio Crameri
    Permission is hereby granted, free of charge, to any person ob- taining a copy of this software and associated documentation files (the ”Software”), to deal in the Software without restric- tion, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is fur- nished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED ”AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

folder = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

__all__ = {'acton', 'bamako', 'batlow', 'berlin', 'bilbao', 'broc', 'buda',
           'cork', 'davos', 'devon', 'grayC', 'hawaii', 'imola', 'lajolla',
           'lapaz', 'lisbon', 'nuuk', 'oleron', 'oslo', 'roma', 'tofino',
           'tokyo', 'turku', 'vik'}

for name in __all__:
    file = os.path.join(folder, name, name + '.txt')
    cm_data = np.loadtxt(file)
    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)

    vars()[name +'_r'] = LinearSegmentedColormap.from_list(name +'_r', np.flip(cm_data))

