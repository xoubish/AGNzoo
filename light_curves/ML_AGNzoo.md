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

By the IPAC Science Platform Team, last edit: Feb 12th, 2024

***


## Goal
Writing these up



## Introduction

Active Galactic Nuclei (AGNs), some of the most powerful sources in the universe, emit a broad range of electromagnetic radiation, from radio waves to gamma rays. Consequently, there is a wide variety of AGN labels depending on the identification/selection scheme and the presence or absence of certain emissions (e.g., Radio loud/quiet, Quasars, Blazars, Seiferts, Changing looks). According to the unified model, this zoo of labels we see depend on a limited number of parameters, namely the viewing angle, the accretion rate, presence or lack of jets, and perhaps the properties of the host/environment (e.g., [Padovani et al. 2017](https://arxiv.org/pdf/1707.07134.pdf)). Here, we collect archival temporal data and labels from the literature to compare how some of these different labels/selection schemes compare.

We use manifold learning and dimensionality reduction to learn the distribution of AGN lightcurves observed with different facilities. We mostly focus on UMAP ([Uniform Manifold Approximation and Projection, McInnes 2020](https://arxiv.org/pdf/1802.03426.pdf)) but also show SOM ([Self organizing Map, Kohonen 1990](https://ieeexplore.ieee.org/document/58325)) examples. The reduced 2D projections from these two unsupervised ML techniques reveal similarities and overlaps of different selection techniques and coloring the projections with various statistical physical properties (e.g., mean brightness, fractional lightcurve variation) is informative of correlations of the selections technique with physics such as AGN variability. Using different parts of the EM in training (or in building the initial higher dimensional manifold) demonstrates how much information if any is in that part of the data for each labeling scheme, for example whether with ZTF optical light curves alone, we can identify sources with variability in WISE near IR bands. These techniques also have a potential for identifying targets of a specific class or characteristic for future follow up observations.

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

from scipy import interpolate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
from scipy.interpolate import interp1d
from tqdm import tqdm
import random

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
colors3 = [
    "#FFD700", # Glowing Yellow, reminiscent of "Starry Night" and "Sunflowers"
    "#6495ED", # Cornflower Blue, reflective of the night sky in "Starry Night"
    "#FF4500", # OrangeRed, for the vibrant tones in "Cafe Terrace at Night"
    "#006400", # DarkGreen, echoing the cypress trees and fields
    "#8B4513", # SaddleBrown, for the earth and branches in many landscapes
    "#DAA520", # Goldenrod, similar to the tones in "Wheatfield with Cypresses"
    "#00008B", # DarkBlue, for the deep night skies
    "#008000", # Green, capturing the vitality of nature scenes
    "#BDB76B", # DarkKhaki, for the muted tones in "The Potato Eaters"
    "#800080", # Purple, reflecting the subtle touches in flowers and clothing
    "#FF6347", # Tomato, for bright accents in paintings like "Red Vineyards at Arles"
    "#4682B4", # SteelBlue, for the serene moments in "Fishing Boats on the Beach"
    "#FA8072", # Salmon, for the soft glow of sunset and sunrise scenes
    "#9ACD32", # YellowGreen, for the lively foliage in many landscapes
    "#40E0D0", # Turquoise, reminiscent of the dynamic strokes in "Irises"
    "#BA55D3", # MediumOrchid, for the playful color in "Almond Blossoms"
    "#7FFF00", # Chartreuse, vibrant and lively for the touches of light
    "#ADD8E6", # LightBlue, for the peaceful skies and distant horizons
]

color4 = ['#3182bd','#6baed6','#9ecae1','#e6550d','#fd8d3c','#fdd0a2','#31a354','#a1d99b', '#c7e9c0', '#756bb1', '#bcbddc', '#dadaeb', '#969696', '#bdbdbd','#d9d9d9']
custom_cmap = LinearSegmentedColormap.from_list("custom_theme", colors2[1:])
```

***


## 1) Loading data
Here we load a parquet file of light curves generated using the multiband_lc notebook. One can build the sample from different sources in the literature and grab the data from archives of interes.

```{code-cell} ipython3
samp = pd.read_csv('data/AGNsample_11Feb24.csv')
redshifts = samp['redshift']

df_lc = pd.read_parquet('data/df_lc_021224.parquet')

#df2 = df[df.index.get_level_values('label') !='64'] # remove 64 for SPIDER only as its too large
#df_lc = update_bitsums(df2,label_num=64) # remove all bitwise sums that had 64 in them

# Filter rows with the specific label
#df4 = df3[df3.index.get_level_values('label') == '64'] # remove 64 for SPIDER only as its too large
# Randomly select rows to drop
#rows_to_drop_indices = np.random.choice(df4.index, size=int(len(df4) * 1), replace=False)
# Drop these rows
#df_lc = df3.drop(rows_to_drop_indices)
```

```{code-cell} ipython3
df_lc
```

```{code-cell} ipython3
from plot_functions import create_figures
_ = create_figures(df_lc = df_lc, # either df_lc (serial call) or parallel_df_lc (parallel call)
                   show_nbr_figures = 1,  # how many plots do you actually want to see?
                   save_output = True ,  # should the resulting plots be saved?
                  )
```

```{code-cell} ipython3
bands_inlc = ['zg','zr','zi','W1','W2']

objects,dobjects,flabels,keeps,zlist = unify_lc(df_lc, redshifts,bands_inlc,xres=160,numplots=3,low_limit_size=5) #nearest neightbor linear interpolation
#objects,dobjects,flabels,keeps,zlist = unify_lc_gp(df_lc,redshifts,bands_inlc,xres=160,numplots=5,low_limit_size=5) #Gaussian process unification

# calculate some basic statistics with a sigmaclipping with width 5sigma
fvar, maxarray, meanarray = stat_bands(objects,dobjects,bands_inlc,sigmacl=5)

# combine different waveband into one array
dat_notnormal = combine_bands(objects,bands_inlc)

# Normalize the combinde array by mean brightness in a waveband after clipping outliers:
datm = normalize_clipmax_objects(dat_notnormal,meanarray,band = 1)

# shuffle data incase the ML routines are sensitive to order
data,fzr,p = shuffle_datalabel(datm,flabels)
fvar_arr,maximum_arr,average_arr = fvar[:,p],maxarray[:,p],meanarray[:,p]
redshift_shuffled = zlist[p]

labc = {}  # Initialize labc to hold indices of each unique label
for index, f in enumerate(fzr):
    lab = translate_bitwise_sum_to_labels(int(f))
    for label in lab:
        if label not in labc:
            labc[label] = []  # Initialize the list for this label if it's not already in labc
        labc[label].append(index)  # Append the current index to the list of indices for this label
```

```{code-cell} ipython3
# Create the figure
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[1,1,1,1])

# Create the subplots
ax0 = fig.add_subplot(gs[0:1, 0:2])  
ax1 = fig.add_subplot(gs[0:1, 2:4])
ax2 = fig.add_subplot(gs[1:2, 0:2])  
ax3 = fig.add_subplot(gs[1:2, 2:4])

objid = df_lc.index.get_level_values('objectid')[:].unique()

seen = Counter()
for (objectid, label), singleobj in df_lc.groupby(level=["objectid", "label"]):
    bitwise_sum = int(label)
    active_labels = translate_bitwise_sum_to_labels(bitwise_sum)
    #active_labels = translate_bitwise_sum_to_labels(label[0])
    seen.update(active_labels)
#changing order of labels in dictionary only for text to be readable on the plot
key_order = ('SDSS_QSO','SPIDER_BL','SPIDER_QSOBL', 'SPIDER_AGNBL',
             'WISE_Variable','Optical_Variable','Galex_Variable','Turn-on', 'Turn-off','TDE')
new_queue = OrderedDict()
for k in key_order:
    new_queue[k] = seen[k]
    
h = ax0.pie(new_queue.values(),labels=new_queue.keys(),autopct=autopct_format(new_queue.values()), textprops={'fontsize': 12},startangle=210,  labeldistance=1.1, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' }, colors=color4[2:])

seen2 = Counter()
for f in fzr:
    active_labels = translate_bitwise_sum_to_labels(int(f))
    seen2.update(active_labels)

new_queue = OrderedDict()
for k in key_order:
    new_queue[k] = seen2[k]


h = ax2.pie(new_queue.values(),labels=new_queue.keys(),autopct=autopct_format(new_queue.values()), textprops={'fontsize': 12},startangle=180,  labeldistance=1.1, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' }, colors=color4[2:])

 
#####################################################################################################
seen = Counter()
seen = df_lc.reset_index().groupby('band').objectid.nunique().to_dict()

cadence = dict((el,[]) for el in seen.keys())
timerange = dict((el,[]) for el in seen.keys())

for (_, band), times in df_lc.reset_index().groupby(["objectid", "band"]).time:
    cadence[band].append(len(times))
    if times.max() - times.min() > 0:
        timerange[band].append(np.round(times.max() - times.min(), 1))


i=0
colorlabel =[0,1,2,0,3,0,6,5,5,8,8,8,8,8,9,9,9]


for el in cadence.keys():
    #print(el,len(cadence[el]),np.mean(cadence[el]),np.std(cadence[el]))
    #print(el,len(timerange[el]),np.mean(timerange[el]),np.std(timerange[el]))
    ax1.scatter(np.mean(cadence[el]),np.mean(timerange[el]),s=seen[el]/2,alpha=0.7,c=color4[colorlabel[i]])
    ax1.errorbar(np.mean(cadence[el]),np.mean(timerange[el]),label=el,yerr=np.std(timerange[el]),xerr=np.std(cadence[el]),alpha=0.2,c=color4[colorlabel[i]])
    #print(el,np.mean(cadence[el]))

    i+=1
    
ax1.annotate('ZTF', # text to display
             (110, 1300),        # text location
             size=12, rotation=40 )
ax1.annotate('WISE', # text to display
             (20, 3800),        # text location
             size=12, rotation=40 )
ax1.annotate('GAIA', # text to display
             (25, 600),        # text location
             size=12, rotation=40 )
ax1.annotate('Pan-STARRS', # text to display
             (22, 1600),        # text location
             size=12, rotation=40 )
ax1.annotate('IceCube', # text to display
             (1, 1500),        # text location
             size=12, rotation=40 )
ax1.annotate('F814W', # text to display
             (6, 2500),        # text location
             size=12, rotation=40 )
ax1.annotate('Fermi', # text to display
             (1, 30),        # text location
             size=12, rotation=40 )

ax1.set_xlabel(r'$\rm Average\ number\ of\ visits$',size=15)
ax1.set_ylabel(r'$\rm Average\ baseline\ (days)$',size=15)
ax1.set_xscale('log')

#####################################################################################################
u = (samp['Fermi_Blazars']!=1)
ax3.hist(redshifts[u],label='initial')
#ax3.hist(redshift_shuffled,label='final')
samp = pd.read_csv('data/AGNsample_11Feb24.csv')

for col in range(2,10):
    u = (samp.iloc[:, col]==1)
    ax3.hist(redshifts[u],histtype='step',label=samp.columns[col])
    
ax3.set_xlabel(r'$\rm Redshifts$',size=15)
ax3.set_ylabel(r'$\rm counts$',size=15)
plt.legend()
plt.tight_layout()
plt.savefig('output/sample.png')
```

```{code-cell} ipython3
plt.figure(figsize=(12,8))
markersize=200
mapper = umap.UMAP(n_neighbors=50,min_dist=0.9,metric='manhattan',random_state=20).fit(data)


ax1 = plt.subplot(2,2,3)
ax1.set_title(r'mean brightness',size=20)
cf = ax1.scatter(mapper.embedding_[:,0],mapper.embedding_[:,1],s=markersize,c=np.log10(np.nansum(meanarray,axis=0)),edgecolor='gray')
plt.axis('off')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf,cax=cax)


ax0 = plt.subplot(2,2,4)
ax0.set_title(r'mean fractional variation',size=20)
cf = ax0.scatter(mapper.embedding_[:,0],mapper.embedding_[:,1],s=markersize,c=stretch_small_values_arctan(np.nansum(fvar_arr,axis=0),factor=3),edgecolor='gray')
plt.axis('off')
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf,cax=cax)

ax2 = plt.subplot(2,2,1)
ax2.set_title('sample origin',size=20)
counts = 1
for label, indices in labc.items():
    cf = ax2.scatter(mapper.embedding_[indices,0],mapper.embedding_[indices,1],s=markersize,c = colors[counts],alpha=0.8,edgecolor='gray',label=label)
    counts+=1
plt.legend(fontsize=10)
#plt.colorbar(cf)
plt.axis('off')

ax3 = plt.subplot(2,2,2)
ax3.set_title('redshifts',size=20)
ax3.scatter(mapper.embedding_[:,0],mapper.embedding_[:,1],s=markersize,c = redshift_shuffled,edgecolor='gray')
plt.axis('off')
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf,cax=cax)

plt.tight_layout()
#plt.savefig('umap-ztf.png')
```

```{code-cell} ipython3
# Calculate 2D histogram
hist, x_edges, y_edges = np.histogram2d(mapper.embedding_[:, 0], mapper.embedding_[:, 1], bins=12)
plt.figure(figsize=(15,12))
i=1
laborder = ['SDSS_QSO','WISE_Variable','Optical_Variable','Galex_Variable','SPIDER_AGN','SPIDER_AGNBL','SPIDER_QSOBL','SPIDER_BL','Turn-on','Turn-off','TDE','Fermi_Blazars']
for label in laborder:
    if label in labc:
        indices = labc[label]
        hist_per_cluster, _, _ = np.histogram2d(mapper.embedding_[indices,0], mapper.embedding_[indices,1], bins=(x_edges, y_edges))
        prob = hist_per_cluster / hist
        plt.subplot(4,4,i)
        plt.title(label)
        plt.contourf(x_edges[:-1], y_edges[:-1], prob.T, levels=20, alpha=0.8,cmap=custom_cmap)
        plt.colorbar()
        plt.axis('off')
        #cf = ax0.scatter(mapper.embedding_[indices,0],mapper.embedding_[indices,1],s=80,alpha=0.5,edgecolor='gray',label=label,c=colors[i-1])
        i+=1
plt.tight_layout()
#plt.savefig('output/umap2-ztfwise-gp.png')
```

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

```{code-cell} ipython3
mapper = umap.UMAP(n_neighbors=10,min_dist=0.9,metric=dtw_distance,random_state=4).fit(data)

plt.scatter(mapper.embedding_[:,0],mapper.embedding_[:,1],s=markersize,c=redshift_shuffled,edgecolor='gray')
plt.colorbar()
```

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

```{code-cell} ipython3
d = np.load('GP_ZTFWISE.npz',allow_pickle=True)
dd = d['data']
dl = d['labc']
```

```{code-cell} ipython3
mapper = umap.UMAP(n_neighbors=50,min_dist=0.9,metric='euclidean',random_state=20).fit(dd)
```

```{code-cell} ipython3
# Calculate 2D histogram
hist, x_edges, y_edges = np.histogram2d(mapper.embedding_[:, 0], mapper.embedding_[:, 1], bins=12)
plt.figure(figsize=(15,12))
i=1
ax0 = plt.subplot(4,4,12)
laborder = ['SDSS_QSO','WISE_Variable','Optical_Variable','Galex_Variable','SPIDER_AGN','SPIDER_AGNBL','SPIDER_QSOBL','SPIDER_BL','Turn-on','Turn-off','TDE']
for label in laborder:
    if label in labc:
        indices = labc[label]
        hist_per_cluster, _, _ = np.histogram2d(mapper.embedding_[indices,0], mapper.embedding_[indices,1], bins=(x_edges, y_edges))
        prob = hist_per_cluster / hist
        plt.subplot(4,4,i)
        plt.title(label)
        plt.contourf(x_edges[:-1], y_edges[:-1], prob.T, levels=20, alpha=0.8,cmap=custom_cmap)
        plt.colorbar()
        plt.axis('off')
        #cf = ax0.scatter(mapper.embedding_[indices,0],mapper.embedding_[indices,1],s=80,alpha=0.5,edgecolor='gray',label=label,c=colors[i-1])
        i+=1
ax0.legend(loc=4,fontsize=7)
ax0.axis('off')
plt.tight_layout()
#plt.savefig('output/umap2-ztfwise-gp.png')
```

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

```{code-cell} ipython3
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
key_order = ('SDSS_QSO','SPIDER_BL','SPIDER_QSOBL', 'SPIDER_AGNBL',
             'WISE_Variable','Optical_Variable','Galex_Variable','Turn-on', 'Turn-off','TDE','Fermi_Blazars')
new_queue = OrderedDict()
for k in key_order:
    new_queue[k] = seen[k]


h = ax0.pie(new_queue.values(),labels=new_queue.keys(),autopct=autopct_format(new_queue.values()), textprops={'fontsize': 12},startangle=210,  labeldistance=1.1, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' }, colors=colors3[:])


# Plot on the second subplot in the first row
seen = Counter()
for b in objid:
    singleobj = df_lc.loc[b,:,:,:]
    label = singleobj.index.unique('label')
    bands = singleobj.loc[label[0],:,:].index.get_level_values('band')[:].unique()
    seen.update(bands)
    
#####################################################################################################

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
colorlabel =[0,1,2,3,3,3,4,4,4,4,4,2,5,5,5,6,7]
for el in cadence.keys():
    #print(el,len(cadence[el]),np.mean(cadence[el]),np.std(cadence[el]))
    #print(el,len(timerange[el]),np.mean(timerange[el]),np.std(timerange[el]))
    ax1.scatter(np.mean(cadence[el]),np.mean(timerange[el]),s=len(timerange[el]),alpha=0.7,c=colors3[colorlabel[i]+1])
    ax1.errorbar(np.mean(cadence[el]),np.mean(timerange[el]),label=el,yerr=np.std(timerange[el]),xerr=np.std(cadence[el]),alpha=0.2,c=colors3[colorlabel[i]+1])
    #print(el,np.mean(cadence[el]))

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
ax1.annotate('F814W', # text to display
             (6, 3000),        # text location
             size=12, rotation=40 )
ax1.annotate('Fermi', # text to display
             (1, 30),        # text location
             size=12, rotation=40 )
#ax1.annotate('GRB', # text to display
#             (1, 10),        # text location
#             size=12, rotation=40 )

#ax1.legend()
ax1.set_xlabel(r'$\rm Average\ number\ of\ visits$',size=15)
ax1.set_ylabel(r'$\rm Average\ baseline\ (days)$',size=15)
ax1.set_xscale('log')

#####################################################################################################
seen = Counter()
for b in objid:
    singleobj = df_lc.loc[b,:,:,:]
    label = singleobj.index.unique('label')
    bands = singleobj.loc[label[0],:,:].index.get_level_values('band')[:].unique()
    seen.update(bands)
    
key_order = ('IceCube','zg', 'zr', 'zi',  'G', 'BP', 'RP', 'panstarrs g',
             'panstarrs r', 'panstarrs i', 'panstarrs z', 'panstarrs y', 'F814W','W1','W2','FERMIGTRIG','SAXGRBMGRB')

#changing order of labels in dictionary only for text to be readable on the plot
new_queue = OrderedDict()
for k in key_order:
    new_queue[k] = seen[k]


tiklabels = ['IceCube','ZTF g','ZTF r','ZTF i','GAIA G', 'GAIA BP', 'GAIA RP',
             'Pan-STARRS g','Pan-STARRS r','Pan-STARRS i','Pan-STARRS z','Pan-STARRS y','F814W','WISE 1','WISE 2','Fermi','GRB']

h = bottom_ax.bar(range(len(tiklabels)), new_queue.values(),color= "#3F51B5")
bottom_ax.set_xticks(range(len(tiklabels)),tiklabels,fontsize=15,rotation=90)
bottom_ax.set_ylabel(r'$\rm number\ of\ lightcurves$',size=15)
plt.tight_layout()

plt.savefig('output/sample.png')
```

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('data/AGNsample_11Feb24.csv')  # Replace 'your_file.csv' with the path to your CSV file
plt.figure(figsize=(10, 6))
for i in range(2, 15):  # Assuming label columns are named as 'label1', 'label2', ..., 'label10'
    label_column = df.columns[i]  # Construct the label column name
    subset_df = df[df[label_column] == 1]  # Filter rows where the current label is 1
    
    # Plot the histogram for the filtered subset
    plt.hist(subset_df['redshift'], bins=10, alpha=0.9, density=True, label=label_column,histtype='step')
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.legend()
```
