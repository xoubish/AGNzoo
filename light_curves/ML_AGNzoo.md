---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# How do AGNs selected with different techniques compare?

By the IPAC Science Platform Team, last edit: Feb 9th, 2024

***


## Learning Goals

```
By the end of this tutorial, you will:

- Work with multi-band lightcurve data
- Learn high dimensional manifold of light curves with UMAPs and SOMs
- Visualize and compare different samples on reduced dimension projections/grids
```


## Introduction

Active Galactic Nuclei (AGNs), some of the most powerful sources in the universe, emit a broad range of electromagnetic radiation, from radio waves to gamma rays. Consequently, there is a wide variety of AGN labels depending on the identification/selection scheme and the presence or absence of certain emissions (e.g., Radio loud/quiet, Quasars, Blazars, Seiferts, Changing looks). According to the unified model, this zoo of labels we see depend on a limited number of parameters, namely the viewing angle, the accretion rate, presence or lack of jets, and perhaps the properties of the host/environment (e.g., [Padovani et al. 2017](https://arxiv.org/pdf/1707.07134.pdf)). Here, we collect archival temporal data and labels from the literature to compare how some of these different labels/selection schemes compare.

We use manifold learning and dimensionality reduction to learn the distribution of AGN lightcurves observed with different facilities. We mostly focus on UMAP ([Uniform Manifold Approximation and Projection, McInnes 2020](https://arxiv.org/pdf/1802.03426.pdf)) but also show SOM ([Self organizing Map, Kohonen 1990](https://ieeexplore.ieee.org/document/58325)) examples. The reduced 2D projections from these two unsupervised ML techniques reveal similarities and overlaps of different selection techniques and coloring the projections with various statistical physical properties (e.g., mean brightness, fractional lightcurve variation) is informative of correlations of the selections technique with physics such as AGN variability. Using different parts of the EM in training (or in building the initial higher dimensional manifold) demonstrates how much information if any is in that part of the data for each labeling scheme, for example whether with ZTF optical light curves alone, we can identify sources with variability in WISE near IR bands. These techniques also have a potential for identifying targets of a specific class or characteristic for future follow up observations.


## Imports
Here are the libraries used in this network. They are also mostly mentioned in the requirements in case you don't have them installed.
- *sys* and *os* to handle file names, paths, and directories
- *numpy*  and *pandas* to handle array functions
- *matplotlib* *pyplot* and *cm* for plotting data
- *astropy.io fits* for accessing FITS files
- *astropy.table Table* for creating tidy tables of the data
- *MultiIndexDFObject*, *ML_utils*, *sample_lc* for reading in and prepreocessing of lightcurve data
- *umap* and *sompy* for manifold learning, dimensionality reduction and visualization

```{code-cell} ipython3
#!pip install -r requirements.txt
import sys
import os
import re
import time

import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
sys.path.append('code_src/')
from data_structures import MultiIndexDFObject
from ML_utils import unify_lc, unify_lc_gp, stat_bands, autopct_format, combine_bands,\
mean_fractional_variation, normalize_mean_objects, normalize_max_objects, \
normalize_clipmax_objects, shuffle_datalabel, dtw_distance, stretch_small_values_arctan, translate_bitwise_sum_to_labels, update_bitsums
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter,OrderedDict

import umap
from sompy import * #using the SOMPY package from https://github.com/sevamoo/SOMPY

import logging

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

plt.style.use('bmh')

colors = ["#3F51B5", "#003153", "#0047AB", "#40826D", "#50C878", "#FFEA00", "#CC7722", "#E34234", "#E30022", "#D68A59", "#8A360F", "#826644", "#5C6BC0", "#002D62", "#0056B3", "#529987", "#66D98B", "#FFED47", "#E69C53", "#F2552C", "#EB4D55", "#E6A875", "#9F5A33", "#9C7C5B", "#2E37FE", "#76D7EA", "#007BA7", "#2E8B57", "#ADDFAD", "#FFFF31"]
colors2 = [
    "#3F51B5",  # Ultramarine Blue
    "#003153",  # Prussian Blue
    "#0047AB",  # Cobalt Blue
    "#40826D",  # Viridian Green
    "#50C878",  # Emerald Green
    "#FFEA00",  # Chrome Yellow
    "#CC7722",  # Yellow Ochre
    "#E34234",  # Vermilion
    "#E30022",  # Cadmium Red
    "#D68A59",  # Raw Sienna
    "#8A360F",  # Burnt Sienna
    "#826644",  # Raw Umber
]

custom_cmap = LinearSegmentedColormap.from_list("custom_theme", colors2[1:])
```

***


## 1) Loading data
Here we load a parquet file of light curves generated using the multiband_lc notebook. One can build the sample from different sources in the literature and grab the data from archives of interes.

```{code-cell} ipython3
#sample_table = Table.read('data/agnsample_feb7.ecsv', format="ascii.ecsv") # if needed, contains coordinates, redshift and all labels
df = pd.read_parquet('data/df_lc_020724.parquet.gzip')

df2 = df[df.index.get_level_values('label') != '64'] # remove 64 for SPIDER only as its too large
df_lc = update_bitsums(df2) # remove all bitwise sums that had 64 in them
```

```{code-cell} ipython3
df_lc
```

```{code-cell} ipython3
from plot_functions import create_figures
_ = create_figures(df_lc = df_lc, # either df_lc (serial call) or parallel_df_lc (parallel call)
                   show_nbr_figures = 4,  # how many plots do you actually want to see?
                   save_output = True ,  # should the resulting plots be saved?
                  )
```

### 1.1) What is in this sample

To effectively undertake machine learning (ML) in addressing a specific question, it's imperative to have a clear understanding of the data we'll be utilizing. This understanding aids in selecting the appropriate ML approach and, critically, allows for informed and necessary data preprocessing. For example whether a normalization is needed, and what band to choose for normalization.
In this particular example, the largest subsamples of AGNs, all with a criteria on redshift (z<1), are from the optical spectra by the [SDSS quasar sample DR16Q](https://www.sdss4.org/dr17/algorithms/qso_catalog/), the value added SDSS spectra from [SPIDERS](https://www.sdss.org/dr18/bhm/programs/spiders/), and a subset of AGNs selected in MIR WISE bands based on their variability ([csv in data folder credit RChary](https://ui.adsabs.harvard.edu/abs/2019AAS...23333004P/abstract)). We also include some smaller samples from the literature to see where they sit compared to the rest of the population and if they are localized on the 2D projection. These include the Changing Look AGNs from the literature (e.g., [LaMassa et al. 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...800..144L/abstract), [Lyu et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...927..227L/abstract), [Hon et al. 2022](https://ui.adsabs.harvard.edu/abs/2022MNRAS.511...54H/abstract)), a sample which showed variability in Galex UV images ([Wasleske et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...933...37W/abstract)), a sample of variable sources identified in optical Palomar observarions ([Baldassare et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...896...10B/abstract)), and the optically variable AGNs in the COSMOS field from a three year program on VLT([De Cicco et al. 2019](https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..33D/abstract)). We also include 30 Tidal Disruption Event coordinates identified from ZTF light curves [Hammerstein et al. 2023](https://iopscience.iop.org/article/10.3847/1538-4357/aca283/meta).
The histogram shows the number of lightcurves which ended up in the multi-index data frame from each of the archive calls in different wavebands/filters.

```{code-cell} ipython3
tiklabels = ['Fermi','ZTF g','ZTF i','ZTF r','WISE 1','WISE 2',
             'Pan-STARRS i','Pan-STARRS r','Pan-STARRS g','Pan-STARRS z','Pan-STARRS y',
            'IceCube','GAIA G', 'GAIA BP', 'GAIA RP','GRB','Kepler','K2','TESS']

# Create the figure
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(3, 3, height_ratios=[2, 2, 2], width_ratios=[1.5,1.5,2])

# Create the subplots
ax0 = fig.add_subplot(gs[0:2, 0:2])  # Top left, taking two-thirds of the width
ax1 = fig.add_subplot(gs[0:2, -1])  # Top right, taking one-third of the width
bottom_ax = fig.add_subplot(gs[2, :])      # Bottom, spanning the full width

objid = df_lc.index.get_level_values('objectid')[:].unique()
seen = Counter()
for b in objid:
    singleobj = df_lc.loc[b,:,:,:]
    label = singleobj.index.unique('label')
    # Translate the bitwise sum back to active labels
    bitwise_sum = int(label[0])  # Convert to integer
    active_labels = translate_bitwise_sum_to_labels(bitwise_sum)
    #active_labels = translate_bitwise_sum_to_labels(label[0])
    seen.update(active_labels)
#changing order of labels in dictionary only for text to be readable on the plot
key_order = ('SDSS_QSO', 'SPIDER_BL','SPIDER_QSOBL', 'SPIDER_AGNBL','SPIDER_AGN',
             'WISE_Variable','Optical_Variable','Galex_Variable','Turn-on', 'Turn-off','TDE')
new_queue = OrderedDict()
for k in key_order:
    new_queue[k] = seen[k]

h = ax0.pie(new_queue.values(),labels=new_queue.keys(),autopct=autopct_format(new_queue.values()), textprops={'fontsize': 12},startangle=210,  labeldistance=1.1, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' }, colors=colors[1:])


# Plot on the second subplot in the first row
seen = Counter()
for b in objid:
    singleobj = df_lc.loc[b,:,:,:]
    label = singleobj.index.unique('label')
    bands = singleobj.loc[label[0],:,:].index.get_level_values('band')[:].unique()
    seen.update(bands)
    

cadence = dict((el,[]) for el in seen.keys())
timerange = dict((el,[]) for el in seen.keys())

for b in objid:
    singleobj = df_lc.loc[b,:,:,:]
    label = singleobj.index.unique('label')
    bband = singleobj.index.unique('band')
    for bb in bband:
        bands = singleobj.loc[label[0],bb,:].index.get_level_values('time')[:]
        #bands.values
        #print(bb,len(bands[:]),np.round(bands[:].max()-bands[:].min(),1))
        cadence[bb].append(len(bands[:]))
        if bands[:].max()-bands[:].min()>0:
            timerange[bb].append(np.round(bands[:].max()-bands[:].min(),1))

i=0
colorlabel = [0,1,1,1,2,2,3,3,3,3,3,4,5,5,5,6,7,7,8]
for el in cadence.keys():
    #print(el,len(cadence[el]),np.mean(cadence[el]),np.std(cadence[el]))
    #print(el,len(timerange[el]),np.mean(timerange[el]),np.std(timerange[el]))
    ax1.scatter(np.mean(cadence[el]),np.mean(timerange[el]),s=len(timerange[el]),alpha=0.7,c=colors[colorlabel[i]+1])
    ax1.errorbar(np.mean(cadence[el]),np.mean(timerange[el]),label=el,yerr=np.std(timerange[el]),xerr=np.std(cadence[el]),alpha=0.2,c=colors[colorlabel[i]+1])
    i+=1
    
ax1.annotate('ZTF', # text to display
             (120, 1300),        # text location
             size=12, rotation=40 )
ax1.annotate('WISE', # text to display
             (20, 3800),        # text location
             size=12, rotation=40 )
ax1.annotate('GAIA', # text to display
             (15, 700),        # text location
             size=12, rotation=40 )
ax1.annotate('Pan-STARRS', # text to display
             (22, 1600),        # text location
             size=12, rotation=40 )
ax1.annotate('IceCube', # text to display
             (1, 1500),        # text location
             size=12, rotation=40 )
ax1.annotate('TESS', # text to display
             (200, 20),        # text location
             size=12, rotation=40 )
ax1.annotate('Kepler', # text to display
             (100, 300),        # text location
             size=12, rotation=40 )

ax1.annotate('GRB', # text to display
             (1, 10),        # text location
             size=12, rotation=40 )

#ax1.legend()
ax1.set_xlabel(r'$\rm Average\ number\ of\ visits$',size=15)
ax1.set_ylabel(r'$\rm Average\ baseline\ (days)$',size=15)
ax1.set_xscale('log')

# Remove the unused subplot (bottom right)
seen = Counter()
for b in objid:
    singleobj = df_lc.loc[b,:,:,:]
    label = singleobj.index.unique('label')
    bands = singleobj.loc[label[0],:,:].index.get_level_values('band')[:].unique()
    seen.update(bands)
    
h = bottom_ax.bar(range(len(tiklabels)), seen.values(),color= "#3F51B5")
bottom_ax.set_xticks(range(len(tiklabels)),tiklabels,fontsize=15,rotation=90)
bottom_ax.set_ylabel(r'$\rm number\ of\ lightcurves$',size=15)
plt.tight_layout()

plt.savefig('output/sample.png')

```

While from the histogram plot we see which bands have the highest number of observed lightcurves, what might matter more in finding/selecting variability or changing look in lightcurves is the cadence and the average baseline of observations. For instance, Panstarrs has a large number of lightcurve detections in our sample, but from the figure above we see that the average number of visits and the baseline for those observations are considerably less than ZTF. WISE also shows the longest baseline of observations which is suitable to finding longer term variability in objects.



## 2) Preprocess data for ML (ZTF bands)

We first look at this sample only in ZTF bands which have the largest number of visits. We start by unifying the time grid of the light curves so oobjects with different start time or number of observations can be compared. We do this by interpolation to a new grid. The choice of the grid resolution and baseline is strictly dependent on the input data, in this case ZTF, to preserve as much as possible all the information from the observations.
The unify_lc, or unify_lc_gp functions do the unification of the lightcurve arrays. For details please see the codes. The time arrays are chosen based on the average duration of observations, with ZTF and WISE covering 1600, 4000 days respectively. We note that we disregard the time of observation of each source, by subtracting the initial time from the array and bringing all lightcurves to the same footing. This has to be taken into account if it influences the science of interest. We then interoplate the time arrays with linear or Gaussian Process regression (unift_lc/ unify_lc_gp respectively). We also remove from the sample objects with less than 5 datapoints in their light curve. We measure basic statistics and combine the tree observed ZTF bands into one longer array as input to dimensionailty reduction after deciding on normalization. We also do a shuffling of the sample to be sure that the separations of different classes by ML are not simply due to the order they are seen in training (in case it is not done by the ML routine itself).

```{code-cell} ipython3
objids = df_lc.index.get_level_values('objectid')[:].unique()
print(objids)
```

```{code-cell} ipython3
translate_bitwise_sum_to_labels(16)
```

```{code-cell} ipython3
from scipy import interpolate

singleobj = df_lc.loc[8238, :, :, :]  # Extract data for the single object
label = singleobj.index.unique('label')  # Get the label of the object
bands = singleobj.loc[label[0], :, :].index.get_level_values('band')[:].unique()  # Extract bands
bands = ['zg','zr','zi','W1','W2']

plt.figure(figsize=(12,6))

for i,band in enumerate(bands):
    if band in ['W1','W2']:
        xout = np.linspace(0,4000,80).reshape(-1, 1) # X array for interpolation
    else:
        xout = np.linspace(0,1600,160).reshape(-1, 1) # X array for interpolation

    kernel_gp = RationalQuadratic(length_scale=1, alpha=0.1)

    band_lc = singleobj.loc[label[0], band, :]  # Extract light curve data for the band

    # Clean data to remove times greater than a threshold (65000)
    band_lc_clean = band_lc[band_lc.index.get_level_values('time') < 65000]
    x, y, dy = np.array(band_lc_clean.index.get_level_values('time') - band_lc_clean.index.get_level_values('time')[0]), np.array(band_lc_clean.flux), np.array(band_lc_clean.err)

    # Sort data based on time
    x2, y2, dy2 = x[np.argsort(x)], y[np.argsort(x)], dy[np.argsort(x)]

    # Check if there are enough points for interpolation
    if len(x2) > 5 and not np.isnan(y2).any():
        # Handle time overlaps in light curves
        n = np.sum(x2 == 0)
        for b in range(1, n):
            x2[::b + 1] = x2[::b + 1] + 1 * 0.001

        # Interpolate the data
        f = interpolate.interp1d(x2, y2, kind='previous', fill_value="extrapolate")
        df = interpolate.interp1d(x2, dy2, kind='previous', fill_value="extrapolate")
        
        gp = GaussianProcessRegressor(kernel=kernel_gp, alpha=dy2**2)
        X = x2.reshape(-1, 1)

        gp.fit(X, y2)

    plt.subplot(2,3,i+1)
    plt.errorbar(x2,y2,yerr=dy,marker='.',linestyle='',markersize=5)
    plt.plot(xout,f(xout),linestyle='--',label='Nearest interpolation')
    ypred,sigma = gp.predict(xout, return_std=True)
    plt.plot(xout,ypred,linestyle='-',label='GP Regression rational quadratic Kernel')
    plt.fill_between(xout.flatten(), ypred - 1.96 * sigma,ypred + 1.96 * sigma, alpha=0.2)
    plt.text(50,0.32,band,size=15)
    plt.xlabel(r'$\rm Time(MJD)$',size=15)
    i+=1
    #plt.ylim([0.1,0.4])


plt.subplot(2,3,1)
plt.ylabel(r'$\rm Flux(mJy)$',size=15)

plt.subplot(2,3,6)
plt.legend(fontsize=8)

plt.tight_layout()
```

```{code-cell} ipython3
from scipy import interpolate

singleobj = df_lc.loc[8238, :, :, :]  # Extract data for the single object
label = singleobj.index.unique('label')  # Get the label of the object
bands = singleobj.loc[label[0], :, :].index.get_level_values('band')[:].unique()  # Extract bands
bands = ['zg','zr','zi']#,'W1','W2']

plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
for i,band in enumerate(bands):
    if band in ['W1','W2']:
        xout = np.linspace(0,4000,80).reshape(-1, 1) # X array for interpolation
    else:
        xout = np.linspace(0,1600,160).reshape(-1, 1) # X array for interpolation

    kernel_gp = RationalQuadratic(length_scale=1, alpha=0.1)

    band_lc = singleobj.loc[label[0], band, :]  # Extract light curve data for the band

    # Clean data to remove times greater than a threshold (65000)
    band_lc_clean = band_lc[band_lc.index.get_level_values('time') < 65000]
    x, y, dy = np.array(band_lc_clean.index.get_level_values('time') - band_lc_clean.index.get_level_values('time')[0]), np.array(band_lc_clean.flux), np.array(band_lc_clean.err)

    # Sort data based on time
    x2, y2, dy2 = x[np.argsort(x)], y[np.argsort(x)], dy[np.argsort(x)]

    # Check if there are enough points for interpolation
    if len(x2) > 5 and not np.isnan(y2).any():
        # Handle time overlaps in light curves
        n = np.sum(x2 == 0)
        for b in range(1, n):
            x2[::b + 1] = x2[::b + 1] + 1 * 0.001

        # Interpolate the data
        f = interpolate.interp1d(x2, y2, kind='previous', fill_value="extrapolate")
        df = interpolate.interp1d(x2, dy2, kind='previous', fill_value="extrapolate")
           
    gpline, = plt.plot(xout,f(xout),linestyle='--')

    line_color = gpline.get_color()  # Get the color of the line
    plt.errorbar(x2,y2,yerr=dy,marker='.',linestyle='',markersize=5,label=band,color=line_color)
    plt.xlabel(r'$\rm Time(MJD)$',size=15)
    i+=1
    #plt.ylim([0.1,0.4])
plt.legend(loc=2)
plt.ylabel(r'$\rm Flux(mJy)$',size=15)

    
plt.subplot(2,2,2)
for i,band in enumerate(bands):
    if band in ['W1','W2']:
        xout = np.linspace(0,4000,80).reshape(-1, 1) # X array for interpolation
    else:
        xout = np.linspace(0,1600,160).reshape(-1, 1) # X array for interpolation

    kernel_gp = RationalQuadratic(length_scale=1, alpha=0.1)
    band_lc = singleobj.loc[label[0], band, :]  # Extract light curve data for the band

    # Clean data to remove times greater than a threshold (65000)
    band_lc_clean = band_lc[band_lc.index.get_level_values('time') < 65000]
    x, y, dy = np.array(band_lc_clean.index.get_level_values('time') - band_lc_clean.index.get_level_values('time')[0]), np.array(band_lc_clean.flux), np.array(band_lc_clean.err)

    # Sort data based on time
    x2, y2, dy2 = x[np.argsort(x)], y[np.argsort(x)], dy[np.argsort(x)]

    # Check if there are enough points for interpolation
    if len(x2) > 5 and not np.isnan(y2).any():
        # Handle time overlaps in light curves
        n = np.sum(x2 == 0)
        for b in range(1, n):
            x2[::b + 1] = x2[::b + 1] + 1 * 0.001

        gp = GaussianProcessRegressor(kernel=kernel_gp, alpha=dy2**2)
        X = x2.reshape(-1, 1)

        gp.fit(X, y2)
    
    ypred,sigma = gp.predict(xout, return_std=True)
    gpline, = plt.plot(xout,ypred,linestyle='-')

    line_color = gpline.get_color()  # Get the color of the line
    plt.errorbar(x2,y2,yerr=dy,marker='.',linestyle='',markersize=5,label=band,color=line_color)

    plt.fill_between(xout.flatten(), ypred - 1.96 * sigma,ypred + 1.96 * sigma, alpha=0.2,color=line_color)
    plt.xlabel(r'$\rm Time(MJD)$',size=15)
    i+=1
    #plt.ylim([0.1,0.4])

plt.subplot(2,2,3)
for i,band in enumerate(['W1','W2']):
    xout = np.linspace(0,4000,80).reshape(-1, 1) # X array for interpolation
    kernel_gp = RationalQuadratic(length_scale=1, alpha=0.1)
    band_lc = singleobj.loc[label[0], band, :]  # Extract light curve data for the band

    band_lc_clean = band_lc[band_lc.index.get_level_values('time') < 65000]
    x, y, dy = np.array(band_lc_clean.index.get_level_values('time') - band_lc_clean.index.get_level_values('time')[0]), np.array(band_lc_clean.flux), np.array(band_lc_clean.err)
    x2, y2, dy2 = x[np.argsort(x)], y[np.argsort(x)], dy[np.argsort(x)]

    # Check if there are enough points for interpolation
    if len(x2) > 5 and not np.isnan(y2).any():
        # Handle time overlaps in light curves
        n = np.sum(x2 == 0)
        for b in range(1, n):
            x2[::b + 1] = x2[::b + 1] + 1 * 0.001

        # Interpolate the data
        f = interpolate.interp1d(x2, y2, kind='previous', fill_value="extrapolate")
        df = interpolate.interp1d(x2, dy2, kind='previous', fill_value="extrapolate")
        

    gpline, = plt.plot(xout,f(xout),linestyle='-')
    line_color = gpline.get_color()  # Get the color of the line
    plt.errorbar(x2,y2,yerr=dy,marker='.',linestyle='',markersize=5,label=band,color=line_color)
    #plt.fill_between(xout.flatten(), f(xout) - 1.96 * sigma,ypred + 1.96 * sigma, alpha=0.2,color=line_color)
    plt.xlabel(r'$\rm Time(MJD)$',size=15)
    i+=1
plt.ylabel(r'$\rm Flux(mJy)$',size=15)
plt.legend(loc=2)

plt.subplot(2,2,4)
for i,band in enumerate(['W1','W2']):
    xout = np.linspace(0,4000,80).reshape(-1, 1) # X array for interpolation
    kernel_gp = RationalQuadratic(length_scale=1, alpha=0.1)
    band_lc = singleobj.loc[label[0], band, :]  # Extract light curve data for the band

    band_lc_clean = band_lc[band_lc.index.get_level_values('time') < 65000]
    x, y, dy = np.array(band_lc_clean.index.get_level_values('time') - band_lc_clean.index.get_level_values('time')[0]), np.array(band_lc_clean.flux), np.array(band_lc_clean.err)
    x2, y2, dy2 = x[np.argsort(x)], y[np.argsort(x)], dy[np.argsort(x)]

    # Check if there are enough points for interpolation
    if len(x2) > 5 and not np.isnan(y2).any():
        # Handle time overlaps in light curves
        n = np.sum(x2 == 0)
        for b in range(1, n):
            x2[::b + 1] = x2[::b + 1] + 1 * 0.001

        # Interpolate the data
        f = interpolate.interp1d(x2, y2, kind='previous', fill_value="extrapolate")
        df = interpolate.interp1d(x2, dy2, kind='previous', fill_value="extrapolate")
        
        gp = GaussianProcessRegressor(kernel=kernel_gp, alpha=dy2**2)
        X = x2.reshape(-1, 1)
        gp.fit(X, y2)
    
    ypred,sigma = gp.predict(xout, return_std=True)
    gpline, = plt.plot(xout,ypred,linestyle='-')
    line_color = gpline.get_color()  # Get the color of the line
    plt.errorbar(x2,y2,yerr=dy,marker='.',linestyle='',markersize=5,label=band,color=line_color)
    plt.fill_between(xout.flatten(), ypred - 1.96 * sigma,ypred + 1.96 * sigma, alpha=0.2,color=line_color)
    plt.xlabel(r'$\rm Time(MJD)$',size=15)
    i+=1

plt.tight_layout()
```

```{code-cell} ipython3
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
from scipy.interpolate import interp1d
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def unified_lc_interpolation(df_lc, bands_inlc=['zr', 'zi', 'zg'], xres=160, numplots=1, low_limit_size=5):
    x_ztf = np.linspace(0, 1600, xres).reshape(-1, 1)
    x_wise = np.linspace(0, 4000, xres).reshape(-1, 1)
    objids = df_lc.index.get_level_values('objectid')[:].unique()

    kernel_gp = RationalQuadratic(length_scale=0.1, alpha=0.1)
    kernel_wise = 1.0 * RBF(length_scale=200)

    printcounter = 0
    objects, dobjects, flabels, keeps = [], [], [], []
    
    for keepindex, obj in tqdm(enumerate(objids)):
        singleobj = df_lc.loc[obj]
        label = singleobj.index.unique('label')
        bands = singleobj.loc[label[0]].index.get_level_values('band').unique()

        keepobj = True
        obj_newy, obj_newdy = [], []

        for l, band in enumerate(bands_inlc):
            if band not in bands:
                keepobj = False
                break

            band_lc = singleobj.loc[(label[0], band)]
            x, y, dy = band_lc.index.get_level_values('time'), band_lc['flux'], band_lc['err']
            x2, y2, dy2 = x[np.argsort(x)], y[np.argsort(x)], dy[np.argsort(x)]

            if len(x2) <= low_limit_size:
                keepobj = False
                break

            X = x2.reshape(-1, 1)

            try:
                if band in ['W1', 'W2']:
                    gp = GaussianProcessRegressor(kernel=kernel_wise, alpha=dy2**2)
                    target_x = x_wise
                else:
                    gp = GaussianProcessRegressor(kernel=kernel_gp, alpha=dy2**2)
                    target_x = x_ztf
                
                gp.fit(X, y2)
                y_pred, sigma = gp.predict(target_x, return_std=True)

                if np.any(np.isnan(y_pred)) or len(y_pred) == 0:  # Check for unsatisfactory GP results
                    raise ValueError("GP interpolation failed or is unsatisfactory.")

            except (ValueError, np.linalg.LinAlgError):
                # Fallback to linear interpolation if GP fails or results are unsatisfactory
                f = interp1d(x2, y2, kind='nearest', fill_value="extrapolate", bounds_error=False)
                y_pred = f(target_x.flatten())

            obj_newy.append(y_pred)
            if printcounter < numplots:
                plt.figure(figsize=(15, 5))
                plt.errorbar(x2, y2, dy2, capsize=1.0, marker='.', linestyle='', label=f'{band} data')
                plt.plot(target_x, y_pred, '--', label=f'{band} interpolation')
                plt.legend()
                plt.xlabel('Time')
                plt.ylabel('Flux')
                plt.title(f'Object {obj} - {band} Band')
                plt.show()

        if keepobj:
            objects.append(obj_newy)
            dobjects.append(obj_newdy)  # Placeholder for error values if necessary
            flabels.append(label[0])
            keeps.append(keepindex)
            printcounter += 1

    return np.array(objects), np.array(dobjects), flabels, keeps
```

```{code-cell} ipython3
bands_inlc = ['zg','zr','zi']

#objects,dobjects,flabels,keeps = unify_lc(df_lc,bands_inlc,xres=160,numplots=5,low_limit_size=5) #nearest neightbor linear interpolation
objects,dobjects,flabels,keeps = unified_lc_interpolation(df_lc,bands_inlc,xres=160,numplots=5,low_limit_size=5) #Gaussian process unification

## keeps can be used as index of objects that are kept in "objects" from
##the initial "df_lc", in case information about some properties of samplen(e.g., redshifts) is of interest this array of indecies would be helpful

# calculate some basic statistics with a sigmaclipping with width 5sigma
fvar, maxarray, meanarray = stat_bands(objects,dobjects,bands_inlc,sigmacl=5)

# combine different waveband into one array
dat_notnormal = combine_bands(objects,bands_inlc)

# Normalize the combinde array by maximum of brightness in a waveband after clipping outliers:
dat = normalize_clipmax_objects(dat_notnormal,maxarray,band = 1)

# Normalize the combinde array by mean brightness in a waveband after clipping outliers:
datm = normalize_clipmax_objects(dat_notnormal,meanarray,band = 1)

# shuffle data incase the ML routines are sensitive to order
data,fzr,p = shuffle_datalabel(dat,flabels)
fvar_arr,maximum_arr,average_arr = fvar[:,p],maxarray[:,p],meanarray[:,p]

labc = {}  # Initialize labc to hold indices of each unique label
for index, f in enumerate(fzr):
    lab = translate_bitwise_sum_to_labels(int(f))
    for label in lab:
        if label not in labc:
            labc[label] = []  # Initialize the list for this label if it's not already in labc
        labc[label].append(index)  # Append the current index to the list of indices for this label
```

The combination of the tree bands into one longer arrays in order of increasing wavelength, can be seen as providing both the SED shape as well as variability in each from the light curve. Figure below demonstrates this as well as our normalization choice. We normalize the data in ZTF R band as it has a higher average numbe of visits compared to G and I band. We remove outliers before measuring the mean and max of the light curve and normalizing by it. This normalization can be skipped if one is mearly interested in comparing brightnesses of the data in this sample, but as dependence on flux is strong to look for variability and compare shapes of light curves a normalization helps.

```{code-cell} ipython3
r = np.random.randint(np.shape(dat)[1])
plt.figure(figsize=(18,4))
plt.subplot(1,3,1)

for i,l in enumerate(bands_inlc):
    s = int(np.shape(dat)[1]/len(bands_inlc))
    first = int(i*s)
    last = first+s
    plt.plot(np.linspace(first,last,s),dat_notnormal[r,first:last],'o',linestyle='--',label=l)
plt.xlabel(r'Time_[w1,w2,w3]',size=15)
plt.ylabel(r'Flux ($\mu Jy$)',size=15)
plt.legend(loc=2)

plt.subplot(1,3,2)
for i,l in enumerate(bands_inlc):
    s = int(np.shape(dat)[1]/len(bands_inlc))
    first = int(i*s)
    last = first+s
    plt.plot(np.linspace(first,last,s),dat[r,first:last],'o',linestyle='--',label=l)
plt.xlabel(r'Time_[w1,w2,w3]',size=15)
plt.ylabel(r'Normalized Flux (max r band)',size=15)

plt.subplot(1,3,3)
for i,l in enumerate(bands_inlc):
    s = int(np.shape(dat)[1]/len(bands_inlc))
    first = int(i*s)
    last = first+s
    plt.plot(np.linspace(first,last,s),datm[r,first:last],'o',linestyle='--',label=l)
plt.xlabel(r'Time_[w1,w2,w3]',size=15)
plt.ylabel(r'Normalized Flux (mean r band)',size=15)
```

## 3) Learn the Manifold


Now we can train a UMAP with the processed data vectors above. Different choices for the number of neighbors, minimum distance and metric can be made and a parameter space can be explored. We show here our preferred combination given this data. We choose manhattan distance (also called [the L1 distance](https://en.wikipedia.org/wiki/Taxicab_geometry)) as it is optimal for the kind of grid we interpolated on, for instance we want the distance to not change if there are observations missing. Another metric appropriate for our purpose in time domain analysis is Dynamic Time Warping ([DTW](https://en.wikipedia.org/wiki/Dynamic_time_warping)), which is insensitive to a shift in time. This is helpful as we interpolate the observations onto a grid starting from time 0 and when discussing variability we care less about when it happens and more about whether and how strong it happened. As the measurement of the DTW distance takes longer compared to the other metrics we show examples here with manhattan and only show one example exploring the parameter space including a DTW metric in the last cell of this notebook.

```{code-cell} ipython3
plt.figure(figsize=(18,6))
markersize=200
mapper = umap.UMAP(n_neighbors=50,min_dist=0.9,metric='manhattan',random_state=20).fit(data)


ax1 = plt.subplot(1,3,2)
ax1.set_title(r'mean brightness',size=20)
cf = ax1.scatter(mapper.embedding_[:,0],mapper.embedding_[:,1],s=markersize,c=np.log10(np.nansum(meanarray,axis=0)),edgecolor='gray')
plt.axis('off')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf,cax=cax)


ax0 = plt.subplot(1,3,3)
ax0.set_title(r'mean fractional variation',size=20)
cf = ax0.scatter(mapper.embedding_[:,0],mapper.embedding_[:,1],s=markersize,c=stretch_small_values_arctan(np.nansum(fvar_arr,axis=0),factor=3),edgecolor='gray')
plt.axis('off')
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf,cax=cax)

ax2 = plt.subplot(1,3,1)
ax2.set_title('sample origin',size=20)
counts = 1
for label, indices in labc.items():
    cf = ax2.scatter(mapper.embedding_[indices,0],mapper.embedding_[indices,1],s=markersize,c = colors[counts],alpha=0.5,edgecolor='gray',label=label)
    counts+=1
plt.legend(fontsize=12)
#plt.colorbar(cf)
plt.axis('off')

plt.tight_layout()
#plt.savefig('umap-ztf.png')
```

The left panel is colorcoded by the origin of the sample. The middle panel shows the sum of mean brightnesses in three bands (arbitrary unit) demonstrating that after normalization we see no correlation with brightness. The panel on the right is color coded by a statistical measure of variability (i.e. the fractional variation [see here](https://ned.ipac.caltech.edu/level5/Sept01/Peterson2/Peter2_1.html)). As with the plotting above it is not easy to see all the data points and correlations in the next two cells measure probability of belonging to each original sample as well as the mean statistical property on an interpolated grid on this reduced 2D projected surface.

```{code-cell} ipython3
# Define a grid
grid_resolution = 12# Number of cells in the grid
x_min, x_max = mapper.embedding_[:, 0].min(), mapper.embedding_[:, 0].max()
y_min, y_max = mapper.embedding_[:, 1].min(), mapper.embedding_[:, 1].max()
x_grid = np.linspace(x_min, x_max, grid_resolution)
y_grid = np.linspace(y_min, y_max, grid_resolution)
x_centers, y_centers = np.meshgrid(x_grid, y_grid)

# Calculate mean property in each grid cell
mean_property1,mean_property2 = np.zeros_like(x_centers),np.zeros_like(x_centers)
propmean=stretch_small_values_arctan(np.nansum(meanarray,axis=0),factor=2)
propfvar=stretch_small_values_arctan(np.nansum(fvar_arr,axis=0),factor=2)
for i in range(grid_resolution - 1):
    for j in range(grid_resolution - 1):
        mask = (
            (mapper.embedding_[:, 0] >= x_grid[i]) &
            (mapper.embedding_[:, 0] < x_grid[i + 1]) &
            (mapper.embedding_[:, 1] >= y_grid[j]) &
            (mapper.embedding_[:, 1] < y_grid[j + 1])
        )
        if np.sum(mask) > 0:
            mean_property1[j, i] = np.mean(propmean[mask])
            mean_property2[j, i] = np.mean(propfvar[mask])


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title('mean brightness')
cf = plt.contourf(x_centers, y_centers, mean_property1, cmap='viridis', alpha=0.9)
plt.axis('off')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf,cax=cax)

plt.subplot(1,2,2)
plt.title('mean fractional variation')
cf = plt.contourf(x_centers, y_centers, mean_property2, cmap='viridis', alpha=0.9)
plt.axis('off')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf,cax=cax)
```

### 3.1) Sample comparison on the UMAP

```{code-cell} ipython3
# Calculate 2D histogram
hist, x_edges, y_edges = np.histogram2d(mapper.embedding_[:, 0], mapper.embedding_[:, 1], bins=12)
plt.figure(figsize=(15,12))
i=1
ax0 = plt.subplot(4,4,12)
for label, indices in sorted(labc.items()):
    hist_per_cluster, _, _ = np.histogram2d(mapper.embedding_[indices,0], mapper.embedding_[indices,1], bins=(x_edges, y_edges))
    prob = hist_per_cluster / hist
    plt.subplot(4,4,i)
    plt.title(label)
    plt.contourf(x_edges[:-1], y_edges[:-1], prob.T, levels=20, alpha=0.8,cmap=custom_cmap)
    plt.colorbar()
    plt.axis('off')
    cf = ax0.scatter(mapper.embedding_[indices,0],mapper.embedding_[indices,1],s=80,alpha=0.5,edgecolor='gray',label=label,c=colors[i-1])
    i+=1
ax0.legend(loc=4,fontsize=7)
ax0.axis('off')
plt.tight_layout()
```

Figure above shows how with ZTF light curves alone we can separate some of these AGN samples, where they have overlaps. We can do a similar exercise with other dimensionality reduction techniques. Below we show two SOMs one with normalized and another with no normalization. The advantage of Umaps to SOMs is that in practice you may change the parameters to separate classes of vastly different data points, as distance is preserved on a umap. On a SOM however only topology of higher dimensions is preserved and not distance hence, the change on the 2d grid does not need to be smooth and from one cell to next there might be larg jumps. On the other hand, an advantage of the SOM is that by definition it has a grid and no need for a posterior interpolation (as we did above) is needed to map more data or to measure probabilities, etc.


### 3.2) Reduced dimensions on a SOM grid

```{code-cell} ipython3
msz0,msz1 = 15,15
sm = sompy.SOMFactory.build(data, mapsize=[msz0,msz1], mapshape='planar', lattice='rect', initialization='pca')
sm.train(n_job=4, shared_memory = 'no')
```

```{code-cell} ipython3
a=sm.bmu_ind_to_xy(sm.project_data(data))
x,y=np.zeros(len(a)),np.zeros(len(a))
k=0
for i in a:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
med_r=np.zeros([msz0,msz1])
fvar_new = stretch_small_values_arctan(np.nansum(fvar_arr,axis=0),factor=1)
for i in range(msz0):
    for j in range(msz1):
        unja=(x==i)&(y==j)
        med_r[i,j]=(np.nanmedian(fvar_new[unja]))


plt.figure(figsize=(18,12))
plt.subplot(3,4,1)
plt.title('SDSS',fontsize=15)
cf=plt.imshow(med_r,origin='lower',cmap='viridis')
plt.axis('off')

def get_indices_by_label(labc, label):
    return labc.get(label, [])

# Example usage
u = labc.get('SDSS_QSO',[])
dsdss = data[u,:]
asdss=sm.bmu_ind_to_xy(sm.project_data(dsdss))
x,y=np.zeros(len(asdss)),np.zeros(len(asdss))
k=0
for i in asdss:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
plt.plot(y,x,'rx',alpha=0.8)

plt.subplot(3,4,2)
plt.title('WISE_Variable',fontsize=15)
cf=plt.imshow(med_r,origin='lower',cmap='viridis')
plt.axis('off')
u = labc.get('WISE_Variable',[])
dwise = data[u,:]
awise=sm.bmu_ind_to_xy(sm.project_data(dwise))
x,y=np.zeros(len(awise)),np.zeros(len(awise))
k=0
for i in awise:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
plt.plot(y,x,'rx',alpha=0.8)

plt.subplot(3,4,3)
plt.title('Optical_Variable',fontsize=15)
cf=plt.imshow(med_r,origin='lower',cmap='viridis')
plt.axis('off')
u = labc.get('Optical_Variable',[])
dcic = data[u,:]
acic=sm.bmu_ind_to_xy(sm.project_data(dcic))
x,y=np.zeros(len(acic)),np.zeros(len(acic))
k=0
for i in acic:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
plt.plot(y,x,'rx',alpha=0.8)


plt.subplot(3,4,4)
plt.title('Galex_Variable',fontsize=15)
cf=plt.imshow(med_r,origin='lower',cmap='viridis')
plt.axis('off')
u = labc.get('Galex_Variable',[])
dcic = data[u,:]
acic=sm.bmu_ind_to_xy(sm.project_data(dcic))
x,y=np.zeros(len(acic)),np.zeros(len(acic))
k=0
for i in acic:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
plt.plot(y,x,'rx',alpha=0.8)

plt.subplot(3,4,5)
plt.title('SPIDER_AGN',fontsize=15)
cf=plt.imshow(med_r,origin='lower',cmap='viridis')
plt.axis('off')
u = labc.get('SPIDER_AGN',[])
dcic = data[u,:]
acic=sm.bmu_ind_to_xy(sm.project_data(dcic))
x,y=np.zeros(len(acic)),np.zeros(len(acic))
k=0
for i in acic:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
plt.plot(y,x,'rx',alpha=0.8)

plt.subplot(3,4,6)
plt.title('SPIDER_AGNBL',fontsize=15)
cf=plt.imshow(med_r,origin='lower',cmap='viridis')
plt.axis('off')
u = labc.get('SPIDER_AGNBL',[])
dcic = data[u,:]
acic=sm.bmu_ind_to_xy(sm.project_data(dcic))
x,y=np.zeros(len(acic)),np.zeros(len(acic))
k=0
for i in acic:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
plt.plot(y,x,'rx',alpha=0.8)

plt.subplot(3,4,7)
plt.title('SPIDER_QSOBL',fontsize=15)
cf=plt.imshow(med_r,origin='lower',cmap='viridis')
plt.axis('off')
u = labc.get('SPIDER_QSOBL',[])
dcic = data[u,:]
acic=sm.bmu_ind_to_xy(sm.project_data(dcic))
x,y=np.zeros(len(acic)),np.zeros(len(acic))
k=0
for i in acic:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
plt.plot(y,x,'rx',alpha=0.8)

plt.subplot(3,4,8)
plt.title('SPIDER_BL',fontsize=15)
cf=plt.imshow(med_r,origin='lower',cmap='viridis')
plt.axis('off')
u = labc.get('SPIDER_BL',[])
dcic = data[u,:]
acic=sm.bmu_ind_to_xy(sm.project_data(dcic))
x,y=np.zeros(len(acic)),np.zeros(len(acic))
k=0
for i in acic:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
plt.plot(y,x,'rx',alpha=0.8)

plt.subplot(3,4,9)
plt.title('Turn-off',fontsize=15)
cf=plt.imshow(med_r,origin='lower',cmap='viridis')
plt.axis('off')
u = labc.get('Turn-off',[])
dcic = data[u,:]
acic=sm.bmu_ind_to_xy(sm.project_data(dcic))
x,y=np.zeros(len(acic)),np.zeros(len(acic))
k=0
for i in acic:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
plt.plot(y,x,'rx',alpha=0.8)


plt.subplot(3,4,10)
plt.title('Turn-on',fontsize=15)
cf=plt.imshow(med_r,origin='lower',cmap='viridis')
plt.axis('off')
u = labc.get('Turn-on',[])
dcic = data[u,:]
acic=sm.bmu_ind_to_xy(sm.project_data(dcic))
x,y=np.zeros(len(acic)),np.zeros(len(acic))
k=0
for i in acic:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
plt.plot(y,x,'rx',alpha=0.8)

plt.subplot(3,4,11)
plt.title('TDE',fontsize=15)
cf=plt.imshow(med_r,origin='lower',cmap='viridis')
plt.axis('off')
u = labc.get('TDE',[])
dcic = data[u,:]
acic=sm.bmu_ind_to_xy(sm.project_data(dcic))
x,y=np.zeros(len(acic)),np.zeros(len(acic))
k=0
for i in acic:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
plt.plot(y,x,'rx',alpha=0.8)

plt.tight_layout()
```

The above SOMs are colored by the mean fractional variation of the lightcurves in all bands (a measure of AGN variability). The crosses are different samples mapped to the trained SOM to see if they are distinguishable on a normalized lightcurve som.

```{code-cell} ipython3
# shuffle data incase the ML routines are sensitive to order
data,fzr,p = shuffle_datalabel(dat_notnormal,flabels)
fvar_arr,maximum_arr,average_arr = fvar[:,p],maxarray[:,p],meanarray[:,p]
labc = {}  # Initialize labc to hold indices of each unique label
for index, f in enumerate(fzr):
    lab = translate_bitwise_sum_to_labels(int(f))
    for label in lab:
        if label not in labc:
            labc[label] = []  # Initialize the list for this label if it's not already in labc
        labc[label].append(index)  # Append the current index to the list of indices for this label
        
sm = sompy.SOMFactory.build(data, mapsize=[msz0,msz1], mapshape='planar', lattice='rect', initialization='pca')
sm.train(n_job=4, shared_memory = 'no')
```

```{code-cell} ipython3
a=sm.bmu_ind_to_xy(sm.project_data(data))
x,y=np.zeros(len(a)),np.zeros(len(a))
k=0
for i in a:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
med_r=np.zeros([msz0,msz1])
fvar_new = stretch_small_values_arctan(np.nansum(fvar_arr,axis=0),factor=1)
for i in range(msz0):
    for j in range(msz1):
        unja=(x==i)&(y==j)
        med_r[i,j]=(np.nanmedian(fvar_new[unja]))


plt.figure(figsize=(18,8))
plt.subplot(1,4,1)
plt.title('SDSS',fontsize=15)
cf=plt.imshow(med_r,origin='lower',cmap='viridis')
plt.axis('off')

def get_indices_by_label(labc, label):
    return labc.get(label, [])

# Example usage
u = labc.get('SDSS_QSO',[])
dsdss = data[u,:]
asdss=sm.bmu_ind_to_xy(sm.project_data(dsdss))
x,y=np.zeros(len(asdss)),np.zeros(len(asdss))
k=0
for i in asdss:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
plt.plot(y,x,'rx',alpha=0.5)

plt.subplot(1,4,2)
plt.title('WISE_Variable',fontsize=15)
cf=plt.imshow(med_r,origin='lower',cmap='viridis')
plt.axis('off')
u = labc.get('WISE_Variable',[])
dwise = data[u,:]
awise=sm.bmu_ind_to_xy(sm.project_data(dwise))
x,y=np.zeros(len(awise)),np.zeros(len(awise))
k=0
for i in awise:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
plt.plot(y,x,'rx',alpha=0.5)

plt.subplot(1,4,3)
plt.title('Turn-on',fontsize=15)
cf=plt.imshow(med_r,origin='lower',cmap='viridis')
plt.axis('off')
u = labc.get('Turn-on',[])
dcic = data[u,:]
acic=sm.bmu_ind_to_xy(sm.project_data(dcic))
x,y=np.zeros(len(acic)),np.zeros(len(acic))
k=0
for i in acic:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
plt.plot(y,x,'rx',alpha=0.5)


plt.subplot(1,4,4)
plt.title('Turn-off',fontsize=15)
cf=plt.imshow(med_r,origin='lower',cmap='viridis')
plt.axis('off')
u = labc.get('Turn-off',[])
dcic = data[u,:]
acic=sm.bmu_ind_to_xy(sm.project_data(dcic))
x,y=np.zeros(len(acic)),np.zeros(len(acic))
k=0
for i in acic:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
plt.plot(y,x,'rx',alpha=0.5)

plt.tight_layout()
```

skipping the normalization of lightcurves, further separates turn on/off CLAGNs when looking at ZTF lightcurves only.


## 4) Repeating the above, this time with ZTF + WISE manifold

```{code-cell} ipython3
bands_inlc = ['zg','zr','zi','W1','W2']
objects,dobjects,flabels,keeps = unify_lc_gp(df_lc,bands_inlc,xres=30,numplots=10)
# calculate some basic statistics
fvar, maxarray, meanarray = stat_bands(objects,dobjects,bands_inlc)
dat_notnormal = combine_bands(objects,bands_inlc)
dat = normalize_clipmax_objects(dat_notnormal,maxarray,band = -1)
data,fzr,p = shuffle_datalabel(dat,flabels)
fvar_arr,maximum_arr,average_arr = fvar[:,p],maxarray[:,p],meanarray[:,p]

labc = {}  # Initialize labc to hold indices of each unique label
for index, f in enumerate(fzr):
    lab = translate_bitwise_sum_to_labels(int(f))
    for label in lab:
        if label not in labc:
            labc[label] = []  # Initialize the list for this label if it's not already in labc
        labc[label].append(index)  # Append the current index to the list of indices for this label
```

```{code-cell} ipython3
plt.figure(figsize=(18,6))
markersize=200
mapper = umap.UMAP(n_neighbors=50,min_dist=0.9,metric='manhattan',random_state=4).fit(data)
#mapper = umap.UMAP(n_neighbors=50,min_dist=0.9,metric=dtw_distance,random_state=20).fit(data) #this distance takes long


ax1 = plt.subplot(1,3,2)
ax1.set_title(r'mean brightness',size=20)
cf = ax1.scatter(mapper.embedding_[:,0],mapper.embedding_[:,1],s=markersize,c=np.log10(np.nansum(meanarray,axis=0)),edgecolor='gray')
plt.axis('off')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf,cax=cax)


ax0 = plt.subplot(1,3,3)
ax0.set_title(r'mean fractional variation',size=20)
cf = ax0.scatter(mapper.embedding_[:,0],mapper.embedding_[:,1],s=markersize,c=stretch_small_values_arctan(np.nansum(fvar_arr,axis=0),factor=3),edgecolor='gray')
plt.axis('off')
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf,cax=cax)

ax2 = plt.subplot(1,3,1)
ax2.set_title('sample origin',size=20)
counts = 1
for label, indices in labc.items():
    cf = ax2.scatter(mapper.embedding_[indices,0],mapper.embedding_[indices,1],s=markersize,c = colors[counts],alpha=0.5,edgecolor='gray',label=label)
    counts+=1
plt.legend(fontsize=12)
#plt.colorbar(cf)
plt.axis('off')

plt.tight_layout()
```

```{code-cell} ipython3
# Calculate 2D histogram
hist, x_edges, y_edges = np.histogram2d(mapper.embedding_[:, 0], mapper.embedding_[:, 1], bins=12)
plt.figure(figsize=(15,12))
i=1
ax0 = plt.subplot(4,4,12)
for label, indices in sorted(labc.items()):
    hist_per_cluster, _, _ = np.histogram2d(mapper.embedding_[indices,0], mapper.embedding_[indices,1], bins=(x_edges, y_edges))
    prob = hist_per_cluster / hist
    plt.subplot(4,4,i)
    plt.title(label)
    plt.contourf(x_edges[:-1], y_edges[:-1], prob.T, levels=20, alpha=0.8,cmap=custom_cmap)
    plt.colorbar()
    plt.axis('off')
    cf = ax0.scatter(mapper.embedding_[indices,0],mapper.embedding_[indices,1],s=80,alpha=0.5,edgecolor='gray',label=label,c=colors[i-1])
    i+=1
ax0.legend(loc=4,fontsize=7)
ax0.axis('off')
plt.tight_layout()
```

## 5) Wise bands alone

```{code-cell} ipython3
bands_inlcw = ['W1','W2']
objectsw,dobjectsw,flabelsw,keepsw = unify_lc(df_lc,bands_inlc,xres=30)
# calculate some basic statistics
fvarw, maxarrayw, meanarrayw = stat_bands(objectsw,dobjectsw,bands_inlcw)
dat_notnormalw = combine_bands(objects,bands_inlcw)
datw = normalize_clipmax_objects(dat_notnormalw,maxarrayw,band = -1)
dataw,fzrw,pw = shuffle_datalabel(datw,flabelsw)
fvar_arrw,maximum_arrw,average_arrw = fvarw[:,pw],maxarrayw[:,pw],meanarrayw[:,pw]

labcw = {}  # Initialize labc to hold indices of each unique label
for index, f in enumerate(fzrw):
    lab = translate_bitwise_sum_to_labels(int(f))
    for label in lab:
        if label not in labcw:
            labcw[label] = []  # Initialize the list for this label if it's not already in labc
        labcw[label].append(index)  # Append the current index to the list of indices for this label
```

```{code-cell} ipython3
plt.figure(figsize=(18,6))
markersize=200
mapp = umap.UMAP(n_neighbors=50,min_dist=0.9,metric='manhattan',random_state=20).fit(dataw)


ax1 = plt.subplot(1,3,2)
ax1.set_title(r'mean brightness',size=20)
cf = ax1.scatter(mapp.embedding_[:,0],mapp.embedding_[:,1],s=markersize,c=np.log10(np.nansum(meanarrayw,axis=0)),edgecolor='gray')
plt.axis('off')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf,cax=cax)


ax0 = plt.subplot(1,3,3)
ax0.set_title(r'mean fractional variation',size=20)
cf = ax0.scatter(mapp.embedding_[:,0],mapp.embedding_[:,1],s=markersize,c=stretch_small_values_arctan(np.nansum(fvar_arrw,axis=0),factor=3),edgecolor='gray')
plt.axis('off')
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf,cax=cax)

ax2 = plt.subplot(1,3,1)
ax2.set_title('sample origin',size=20)
counts = 1
for label, indices in labcw.items():
    cf = ax2.scatter(mapp.embedding_[indices,0],mapp.embedding_[indices,1],s=markersize,c = colors[counts],alpha=0.5,edgecolor='gray',label=label)
    counts+=1
plt.legend(fontsize=12)
#plt.colorbar(cf)
plt.axis('off')

plt.tight_layout()
```

```{code-cell} ipython3
# Calculate 2D histogram
hist, x_edges, y_edges = np.histogram2d(mapp.embedding_[:, 0], mapp.embedding_[:, 1], bins=12)
plt.figure(figsize=(15,12))
i=1
ax0 = plt.subplot(4,4,12)
for label, indices in sorted(labcw.items()):
    hist_per_cluster, _, _ = np.histogram2d(mapp.embedding_[indices,0], mapp.embedding_[indices,1], bins=(x_edges, y_edges))
    prob = hist_per_cluster / hist
    plt.subplot(4,4,i)
    plt.title(label)
    plt.contourf(x_edges[:-1], y_edges[:-1], prob.T, levels=20, alpha=0.8,cmap=custom_cmap)
    plt.colorbar()
    plt.axis('off')
    cf = ax0.scatter(mapp.embedding_[indices,0],mapp.embedding_[indices,1],s=80,alpha=0.5,edgecolor='gray',label=label,c=colors[i-1])
    i+=1
ax0.legend(loc=4,fontsize=7)
ax0.axis('off')
plt.tight_layout()
```

## 6) UMAP with different metrics/distances on ZTF+WISE
DTW takes a bit longer compared to other metrics.

```{code-cell} ipython3
plt.figure(figsize=(12,10))
markersize=200

mapper = umap.UMAP(n_neighbors=50,min_dist=0.9,metric='euclidean',random_state=20).fit(data)
ax0 = plt.subplot(2,2,1)
ax0.set_title(r'Euclidean Distance, min_d=0.9, n_neighbors=50',size=12)
for label, indices in (labc.items()):
     cf = ax0.scatter(mapper.embedding_[indices,0],mapper.embedding_[indices,1],s=80,alpha=0.5,edgecolor='gray',label=label)
plt.axis('off')

mapper = umap.UMAP(n_neighbors=50,min_dist=0.9,metric='manhattan',random_state=20).fit(data)
ax0 = plt.subplot(2,2,2)
ax0.set_title(r'Manhattan Distance, min_d=0.9, n_neighbors=50',size=12)
for label, indices in (labc.items()):
     cf = ax0.scatter(mapper.embedding_[indices,0],mapper.embedding_[indices,1],s=80,alpha=0.5,edgecolor='gray',label=label)
plt.axis('off')


mapperg = umap.UMAP(n_neighbors=50,min_dist=0.9,metric=dtw_distance,random_state=20).fit(data) #this distance takes long
ax2 = plt.subplot(2,2,3)
ax2.set_title(r'DTW Distance, min_d=0.9,n_neighbors=50',size=12)
for label, indices in (labc.items()):
     cf = ax2.scatter(mapper.embedding_[indices,0],mapper.embedding_[indices,1],s=80,alpha=0.5,edgecolor='gray',label=label)
plt.axis('off')


mapper = umap.UMAP(n_neighbors=50,min_dist=0.1,metric='manhattan',random_state=20).fit(data)
ax0 = plt.subplot(2,2,4)
ax0.set_title(r'Manhattan Distance, min_d=0.1, n_neighbors=50',size=12)
for label, indices in (labc.items()):
     cf = ax0.scatter(mapper.embedding_[indices,0],mapper.embedding_[indices,1],s=80,alpha=0.5,edgecolor='gray',label=label)
plt.legend(fontsize=12)
plt.axis('off')
```

## About this Notebook
This notebook is created by the IPAC science platform team as a usecase of ML for time domain astrophysics. For questions contact: shemmati@caltech.edu

**Author:** Shoubaneh Hemmati, Research scientist
**Updated On:** 2024-09-02


## Citations

Parts of this notebook wikk be presented in Hemmati et al. (in prep)

Datasets:
* TBD

Packages:
* [`SOMPY`](https://github.com/sevamoo/SOMPY)
* [`umap`](https://github.com/lmcinnes/umap)



[Top of Page](#top)
