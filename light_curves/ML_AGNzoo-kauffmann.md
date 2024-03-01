---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: root *
  language: python
  name: conda-root-py
---

# How do AGNs selected with different techniques compare?

By the IPAC Science Platform Team, last edit: Feb 29th, 2024

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
from ML_utils import unify_lc, unify_lc_gp,unify_lc_gp_parallel, stat_bands, autopct_format, combine_bands,\
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

color4 = ['#3182bd','#6baed6','#9ecae1','#e6550d','#fd8d3c','#fdd0a2','#31a354','#a1d99b', '#c7e9c0', '#756bb1', '#bcbddc', '#dadaeb', '#969696', '#bdbdbd','#d9d9d9']
custom_cmap = LinearSegmentedColormap.from_list("custom_theme", color4)
```

***


## 1) Loading data
Here we load a parquet file of light curves generated using the multiband_lc notebook. One can build the sample from different sources in the literature and grab the data from archives of interes.

```{code-cell} ipython3
r=0
redshifts, o3lum,o3corr, bpt1,bpt2, rml50, rmu, con, d4n,hda, vdisp = [],[],[],[],[],[],[],[],[],[],[]
with open("data/agn.dat_dr4_release.v2", 'r') as file:
    for line in file:
        parts = line.split()  # Splits the line into parts
        redshifts.append(float(parts[5]))
        o3lum.append(float(parts[6]))
        o3corr.append(float(parts[7]))
        bpt1.append(float(parts[8]))
        bpt2.append(float(parts[9]))
        rml50.append(float(parts[10]))
        rmu.append(float(parts[11]))
        con.append(float(parts[12]))
        d4n.append(float(parts[13]))
        hda.append(float(parts[14]))
        vdisp.append(float(parts[15]))
        r+=1
redshifts, o3lum,o3corr, bpt1,bpt2, rml50, rmu, con, d4n,hda, vdisp = np.array(redshifts), np.array(o3lum),np.array(o3corr), np.array(bpt1),np.array(bpt2), np.array(rml50), np.array(rmu), np.array(con), np.array(d4n),np.array(hda), np.array(vdisp)

df_lc = pd.read_parquet('output/df_lc_kauffmann.parquet')
```

```{code-cell} ipython3
bands_inlc = ['zg','zr','W1','W2']
numobjs = 25000
#objects,dobjects,flabels,keeps,zlist = unify_lc(df_lc, redshifts,bands_inlc,xres=160,numplots=3,low_limit_size=50) #nearest neightbor linear interpolation
#objects,dobjects,flabels,keeps,zlist = unify_lc_gp(df_lc,redshifts,bands_inlc,xres=160,numplots=5,low_limit_size=10) #Gaussian process unification
sample_objids = df_lc.index.get_level_values('objectid').unique()[:numobjs]
df_lc_small = df_lc.loc[sample_objids]
objects,dobjects,flabels,zlist,keeps = unify_lc_gp_parallel(df_lc_small,redshifts[:numobjs],bands_inlc=bands_inlc,xres=120)


# calculate some basic statistics with a sigmaclipping with width 5sigma
fvar, maxarray, meanarray = stat_bands(objects,dobjects,bands_inlc,sigmacl=5)

# combine different waveband into one array
dat_notnormal = combine_bands(objects,bands_inlc)

# Normalize the combinde array by mean brightness in a waveband after clipping outliers:
datm = normalize_clipmax_objects(dat_notnormal,meanarray,band = 0)

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
# Assuming `data` is your numpy array
nan_rows = np.any(np.isnan(data), axis=1)
clean_data = data[~nan_rows]  # Rows without NaNs

redshifts1, o3lum1,o3corr1, bpt11,bpt21, rml501, rmu1, con1, d4n1,hda1, vdisp1= redshifts[keeps], o3lum[keeps],o3corr[keeps], bpt1[keeps],bpt2[keeps], rml50[keeps], rmu[keeps], con[keeps], d4n[keeps],hda[keeps], vdisp[keeps]
redshifts2, o3lum2,o3corr2, bpt12,bpt22, rml502, rmu2, con2, d4n2,hda2, vdisp2 = redshifts1[p], o3lum1[p],o3corr1[p], bpt11[p],bpt21[p], rml501[p], rmu1[p], con1[p], d4n1[p],hda1[p], vdisp1[p]
redshifts3, o3lum3,o3corr3, bpt13,bpt23, rml503, rmu3, con3, d4n3,hda3, vdisp3 = redshifts2[~nan_rows], o3lum2[~nan_rows],o3corr2[~nan_rows], bpt12[~nan_rows],bpt22[~nan_rows], rml502[~nan_rows], rmu2[~nan_rows], con2[~nan_rows], d4n2[~nan_rows],hda2[~nan_rows], vdisp2[~nan_rows]

fvar_arr1,average_arr1 = fvar_arr[:,~nan_rows],average_arr[:,~nan_rows]
```

```{code-cell} ipython3
plt.figure(figsize=(12,5))
markersize=200
mapper = umap.UMAP(n_neighbors=5,min_dist=0.9,metric='manhattan',random_state=20).fit(clean_data)

ax1 = plt.subplot(1,2,1)
ax1.set_title(r'mean brightness',size=20)
cf = ax1.scatter(mapper.embedding_[:,0],mapper.embedding_[:,1],s=markersize,c=np.log10(np.nansum(average_arr1,axis=0)),edgecolor='gray')
plt.axis('off')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf,cax=cax)


ax0 = plt.subplot(1,2,2)
ax0.set_title(r'mean fractional variation',size=20)
cf = ax0.scatter(mapper.embedding_[:,0],mapper.embedding_[:,1],s=markersize,c=stretch_small_values_arctan(np.nansum(fvar_arr1,axis=0),factor=3),edgecolor='gray')
plt.axis('off')
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cf,cax=cax)
plt.tight_layout()
#plt.savefig('umap-ztf.png')
```

```{code-cell} ipython3
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
thiscolor=o3lum3
u = (thiscolor<9) & (thiscolor>3)
plt.title('OIII Luminosity')
plt.scatter(mapper.embedding_[u,0],mapper.embedding_[u,1],s=10,c = thiscolor[u],edgecolor='gray')
plt.axis('off')
plt.colorbar()

plt.subplot(1,2,2)
thiscolor=o3corr3
u = (thiscolor<10) & (thiscolor>3)
plt.title('OIII Luminosity corrected')
plt.scatter(mapper.embedding_[u,0],mapper.embedding_[u,1],s=10,c = thiscolor[u],edgecolor='gray')
plt.axis('off')
plt.colorbar()
```

```{code-cell} ipython3
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
thiscolor=bpt13
u = (thiscolor<1.5) & (thiscolor>-1)
plt.title('BPT1')
plt.scatter(mapper.embedding_[u,0],mapper.embedding_[u,1],s=15,c = thiscolor[u],edgecolor='gray')
plt.axis('off')
plt.colorbar()

plt.subplot(1,2,2)
thiscolor=bpt23
u = (thiscolor<1.) & (thiscolor>-1)
plt.title('BPT2')
plt.scatter(mapper.embedding_[u,0],mapper.embedding_[u,1],s=15,c = thiscolor[u],edgecolor='gray')
plt.axis('off')
plt.colorbar()
```

```{code-cell} ipython3
plt.figure(figsize=(16,4))
plt.subplot(1,3,1)
thiscolor=rml503
u = (thiscolor<13) & (thiscolor>8)
plt.title('Mass 50')
plt.scatter(mapper.embedding_[u,0],mapper.embedding_[u,1],s=15,c = thiscolor[u],edgecolor='gray')
plt.axis('off')
plt.colorbar()

plt.subplot(1,3,2)
thiscolor=rmu3
u = (thiscolor<11) & (thiscolor>7)
plt.title('Mass Surface density')
plt.scatter(mapper.embedding_[u,0],mapper.embedding_[u,1],s=15,c = thiscolor[u],edgecolor='gray')
plt.axis('off')
plt.colorbar()

plt.subplot(1,3,3)
thiscolor=con3
u = (thiscolor<5) & (thiscolor>1)
plt.title('Concentration')
plt.scatter(mapper.embedding_[u,0],mapper.embedding_[u,1],s=15,c = thiscolor[u],edgecolor='gray')
plt.axis('off')
plt.colorbar()
```

```{code-cell} ipython3
plt.figure(figsize=(16,4))
plt.subplot(1,3,1)
thiscolor=d4n3
u = (thiscolor<2.5) & (thiscolor>1)
plt.title('D4000?')
plt.scatter(mapper.embedding_[u,0],mapper.embedding_[u,1],s=15,c = thiscolor[u],edgecolor='gray')
plt.axis('off')
plt.colorbar()

plt.subplot(1,3,2)
thiscolor=hda3
u = (thiscolor<10) & (thiscolor>-10)
plt.title('HDA')
plt.scatter(mapper.embedding_[u,0],mapper.embedding_[u,1],s=15,c = thiscolor[u],edgecolor='gray')
plt.axis('off')
plt.colorbar()

plt.subplot(1,3,3)
thiscolor=vdisp3
u = (thiscolor<400) & (thiscolor>0)
plt.title('Vdispersion')
plt.scatter(mapper.embedding_[u,0],mapper.embedding_[u,1],s=15,c = thiscolor[u],edgecolor='gray')
plt.axis('off')
plt.colorbar()
```

```{code-cell} ipython3
msz0,msz1 = 40,40
sm = sompy.SOMFactory.build(clean_data, mapsize=[msz0,msz1], mapshape='planar', lattice='rect', initialization='pca')
sm.train(n_job=4, shared_memory = 'no')
```

```{code-cell} ipython3
a=sm.bmu_ind_to_xy(sm.project_data(clean_data))
x,y=np.zeros(len(a)),np.zeros(len(a))
k=0
for i in a:
    x[k]=i[0]
    y[k]=i[1]
    k+=1
BPT1_r,BPT2_r=np.zeros([msz0,msz1]),np.zeros([msz0,msz1])
for i in range(msz0):
    for j in range(msz1):
        unja=(x==i)&(y==j)
        BPT1_r[i,j]=(np.nanmedian(bpt13[unja]))
        BPT2_r[i,j]=(np.nanmedian(bpt23[unja]))

plt.figure(figsize=(9,5))
plt.subplot(1,2,1)
cf=plt.imshow(BPT1_r,vmin=-1,vmax=1,origin='lower',cmap='viridis')
plt.axis('off')
plt.subplot(1,2,2)
cf=plt.imshow(BPT2_r,vmin=-1,vmax=1,origin='lower',cmap='viridis')
plt.axis('off')
plt.tight_layout()
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

```

```{code-cell} ipython3

```
