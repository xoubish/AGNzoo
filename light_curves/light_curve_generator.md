---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Make Multiwavelength Light Curves Using Archival Data
***

## Learning Goals    
By the end of this tutorial, you will be able to:  
  &bull; Automatically load a catalog of sources  
  &bull; Automatically & efficiently search NASA and non-NASA resources for light curves at scale  
  &bull; Store & manipulate light curves in a Pandas multiindex dataframe  
  &bull; Plot all light curves on the same plot
 
 
## Introduction:  
 &bull; A user has a sample of interesting targets for which they would like to see a plot of available archival light curves.  We start with a small set of changing look AGN from Yang et al., 2018, which are automatically downloaded. Changing look AGN are cases where the broad emission lines appear or disappear (and not just that the flux is variable). 
 
 &bull; We model light curve plots after van Velzen et al. 2021.  We search through a curated list of time-domain NASA holdings as well as non-NASA sources.  HEASARC catalogs used are Fermi and Beppo-Sax, IRSA catalogs used are ZTF and WISE, and MAST catalogs used are Pan-Starrs, TESS, Kepler, and K2.  Non-NASA sources are Gaia and IceCube. This list is generalized enough to include many types of targets to make this notebook interesting for many types of science.  All of these time-domain archives are searched in an automated and efficient fashion using astroquery, pyvo, pyarrow or APIs.
 
 &bull; Light curve data storage is a tricky problem.  Currently we are using a multi-index Pandas dataframe, as the best existing choice for right now.  One downside is that we need to manually track the units of flux and time instead of relying on an astropy storage scheme which would be able to do some of the units worrying for us (even astropy can't do all magnitude to flux conversions).  Astropy does not currently have a good option for multi-band light curve storage.
 
 &bull; ML work using these time-series light curves is in two neighboring notebooks: ML_AGNzoo and lc_classifier.
 
## Input:
 &bull; choose from a list of known changing look AGN from the literature  
  OR -    
 &bull; input your own sample

## Output:
 &bull; an archival optical + IR + neutrino light curve  
 
## Imports:
 &bull; `acstools` to work with HST magnitude to flux conversion  
 &bull; `astropy` to work with coordinates/units and data structures  
 &bull; `astroquery` to interface with archives APIs  
 &bull; `hpgeom` to locate coordinates in HEALPix space  
 &bull; `lightkurve` to search TESSS, Kepler, and K2 archives  
 &bull; `matplotlib` for plotting  
 &bull; `multiprocessing` to use the power of multiple CPUs to get work done faster  
 &bull; `numpy` for numerical processing  
 &bull; `pandas` for their data structure DataFrame and all the accompanying functions  
 &bull; `pyarrow` to work with Parquet files for WISE and ZTF  
 &bull; `pyvo` for acessing Virtual Observatory(VO) standard data  
 &bull; `requests` to get information from URLs  
 &bull; `scipy` to do statistics  
 &bull; `tqdm` to track progress on long running jobs  
 &bull; `urllib` to handle archive searches with website interface

## Authors:
Jessica Krick, Shoubaneh Hemmati, Andreas Faisst, Troy Raen, Brigitta Sipőcz, Dave Shupe

## Acknowledgements:
Suvi Gezari, Antara Basu-zych, Stephanie LaMassa  
MAST, HEASARC, & IRSA Fornax teams

```{code-cell} ipython3
# Ensure all dependencies are installed
#!pip install -r requirements.txt
```

```{code-cell} ipython3
import multiprocessing as mp
import sys
import time

import astropy.units as u
import pandas as pd
from astropy.table import Table

# local code imports
sys.path.append('code_src/')
from data_structures import MultiIndexDFObject
from gaia_functions import Gaia_get_lightcurve
from HCV_functions import HCV_get_lightcurves
from heasarc_functions import HEASARC_get_lightcurves
from icecube_functions import Icecube_get_lightcurve
from panstarrs import Panstarrs_get_lightcurves
from plot_functions import create_figures
from sample_selection import (clean_sample, get_green_sample, get_hon_sample, get_lamassa_sample, get_lopeznavas_sample,
    get_lyu_sample, get_macleod16_sample, get_macleod19_sample, get_ruan_sample, get_SDSS_sample, get_sheng_sample, get_yang_sample)
from TESS_Kepler_functions import TESS_Kepler_get_lightcurves
# Note: WISE and ZTF data are temporarily located in a non-public AWS S3 bucket. It is automatically
# available from the Fornax SMCE, but will require user credentials for access outside the SMCE.
from WISE_functions import WISE_get_lightcurves
from ztf_functions import ZTF_get_lightcurve
```

## 1. Define the sample
We define here a "gold" sample of spectroscopically confirmed changing look AGN and quasars. This sample includes both objects which change from type 1 to type 2 and also the opposite.  Future studies may want to treat these as seperate objects or seperate QSOs from AGN.  Bibcodes for the samples used are listed next to their functions for reference.  
 
Significant work went into the functions which grab the samples from the papers.  They use Astroquery, NED, SIMBAD, Vizier, and in a few cases grab the tables from the html versions of the paper.  There are trickeries involved in accessing coordinates from tables in the literature. Not every literature table is stored in its entirety in all of these resrources, so be sure to check that your chosen method is actually getting the information that you see in the paper table.  Warning: You will get false results if using NED or SIMBAD on a table that has more rows than are printed in the journal.

```{code-cell} ipython3
# Build up the sample
# Initially set up lists to hold the coordinates and their reference paper name as a label
coords =[]
labels = []

# Choose your own adventure:

#get_lamassa_sample(coords, labels)  #2015ApJ...800..144L
#get_macleod16_sample(coords, labels) #2016MNRAS.457..389M
#get_ruan_sample(coords, labels) #2016ApJ...826..188R
#get_macleod19_sample(coords, labels)  #2019ApJ...874....8M
#get_sheng_sample(coords, labels)  #2020ApJ...889...46S
#get_green_sample(coords, labels)  #2022ApJ...933..180G
#get_lyu_sample(coords, labels)  #z32022ApJ...927..227L
#get_lopeznavas_sample(coords, labels)  #2022MNRAS.513L..57L
#get_hon_sample(coords, labels)  #2022MNRAS.511...54H
get_yang_sample(coords, labels)   #2018ApJ...862..109Y

# Get some "normal" QSOs 
# there are ~500K of these, so choose the number based on
# a balance between speed of running the light curves and whatever 
# the ML algorithms would like to have

#num_normal_QSO = 5000
#get_SDSS_sample(coords, labels, num_normal_QSO)

# Remove duplicates, attach an objectid to the coords,
# convert to astropy table to keep all relevant info together
sample_table = clean_sample(coords, labels)
```

### 1.1 Build your own sample

To build your own sample, you can follow the examples of functions above to grab coordinates from your favorite literature resource, 

or

You can use [astropy's read](https://docs.astropy.org/en/stable/io/ascii/read.html) function to read in an input table
to an [astropy table](https://docs.astropy.org/en/stable/table/)

+++

### 1.2 Write out your sample to disk

At this point you may wish to write out your sample to disk and reuse that in future work sessions, instead of creating it from scratch again.

For the format of the save file, we would suggest to choose from various formats that fully support astropy objects(eg., SkyCoord).  One example that works is Enhanced Character-Separated Values or ['ecsv'](https://docs.astropy.org/en/stable/io/ascii/ecsv.html)

```{code-cell} ipython3
sample_table.write('data/input_sample.ecsv', format='ascii.ecsv', overwrite = True)
```

### 1.3 Load the sample table from disk

Do only this step from this section when you have a previously generated sample table

```{code-cell} ipython3
#sample_table = Table.read('data/input_sample.ecsv', format='ascii.ecsv')
sample_table = Table.read('AGNsample_06Feb24.csv', format='csv')
```

### 1.4 Initialize data structure to hold the light curves

```{code-cell} ipython3
# We wrote our own class for a Pandas MultiIndex [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) for storing the light curves
# This class helps simplify coding of common uses for the DataFrame.
df_lc = MultiIndexDFObject()
```

## 2. Find light curves for these targets in NASA catalogs
We search a curated list of time-domain catalogs from NASA astrophysics archives.  Because each archive is different, and in many cases each catalog is different, each function to access a catalog is necesarily specialized to the location and format of that particular catalog.

+++

### 2.1 HEASARC: FERMI & Beppo SAX
The function to retrieve HEASARC data accesses the HEASARC archive using a pyvo search with a table upload.  This is the fastest way to access data from HEASARC catalogs at scale.  

While these aren't strictly light curves, we would like to track if there are gamma rays detected in advance of any change in the CLAGN light curves. We store these gamma ray detections as single datapoints.  Because gamma ray detections typically have very large error radii, our current technique is to keep matches in the catalogs within some manually selected error radius, currently defaulting to 1 degree for Fermi and 3 degrees for Beppo SAX.  These values are chosen based on a histogram of all values for those catalogs.

```{code-cell} ipython3
start_serial = time.time()  #keep track of all serial archive calls to compare later with parallel archive call time
heasarcstarttime = time.time()

# What is the size of error_radius for the catalogs that we will accept for our cross-matching?
# in degrees; chosen based on histogram of all values for these catalogs
max_fermi_error_radius = str(1.0)  
max_sax_error_radius = str(3.0)

# catalogs to query and their corresponding max error radii
heasarc_catalogs = {"FERMIGTRIG": max_fermi_error_radius, "SAXGRBMGRB": max_sax_error_radius}

# get heasarc light curves in the above curated list of catalogs
df_lc_HEASARC = HEASARC_get_lightcurves(sample_table, heasarc_catalogs)

# add the resulting dataframe to all other archives
df_lc.append(df_lc_HEASARC)

print('heasarc search took:', time.time() - heasarcstarttime, 's')
```

### 2.2 IRSA: ZTF
The function to retrieve ZTF light curves accesses a parquet version of the ZTF catalog stored in the cloud using pyarrow.  This is the fastest way to access the ZTF catalog at scale.  The ZTF [API](https://irsa.ipac.caltech.edu/docs/program_interface/ztf_lightcurve_api.html) is available for small sample searches.  One unique thing about this function is that it has parallelization built in to the function itself.

```{code-cell} ipython3
ZTFstarttime = time.time()

# get ZTF lightcurves
# use the nworkers arg to control the amount of parallelization in the data loading step
df_lc_ZTF = ZTF_get_lightcurve(sample_table, nworkers=6)

# add the resulting dataframe to all other archives
df_lc.append(df_lc_ZTF)

print('ZTF search took:', time.time() - ZTFstarttime, 's')
```

### 2.3 IRSA: WISE

We use the unWISE light curves catalog ([Meisner et al., 2023](https://ui.adsabs.harvard.edu/abs/2023AJ....165...36M/abstract)) which ties together all WISE & NEOWISE 2010 - 2020 epochs.  Specifically it combines all observations at a single epoch to achieve deeper mag limits than individual observations alone.

The function to retrieve WISE light curves accesses an IRSA generated version of the catalog in parquet format being stored in the AWS cloud [Open Data Repository](https://registry.opendata.aws/collab/nasa/)

```{code-cell} ipython3
WISEstarttime = time.time()

bandlist = ['W1', 'W2']  #list of the WISE band names
WISE_radius = 1.0 * u.arcsec
# get WISE light curves
df_lc_WISE = WISE_get_lightcurves(sample_table, WISE_radius, bandlist)

# add the resulting dataframe to all other archives
df_lc.append(df_lc_WISE)

print('WISE search took:', time.time() - WISEstarttime, 's')
```

### 2.4 MAST: Pan-STARRS
The function to retrieve lightcurves from Pan-STARRS currently uses their API; based on this [example](https://ps1images.stsci.edu/ps1_dr2_api.html).  This search is not efficient at scale and we expect it to be replaced in the future.

```{code-cell} ipython3
panstarrsstarttime = time.time()

panstarrs_search_radius = 1.0/3600.0    # search radius = 1 arcsec
# get panstarrs light curves
df_lc_panstarrs = Panstarrs_get_lightcurves(sample_table, panstarrs_search_radius)

# add the resulting dataframe to all other archives
df_lc.append(df_lc_panstarrs)

print('Panstarrs search took:', time.time() - panstarrsstarttime, 's')
```

### 2.5 MAST: TESS, Kepler and K2
The function to retrieve lightcurves from these three missions currently uses the open source package [`lightKurve`](https://docs.lightkurve.org/index.html).  This search is not efficient at scale and we expect it to be replaced in the future.

```{code-cell} ipython3
lightkurvestarttime = time.time()

TESS_search_radius = 1.0  #arcseconds
# get TESS/Kepler/K2 light curves
df_lc_TESS = TESS_Kepler_get_lightcurves(sample_table, TESS_search_radius)

# add the resulting dataframe to all other archives
df_lc.append(df_lc_TESS)

print('TESS/Kepler/K2 search took:', time.time() - lightkurvestarttime, 's')

# LightKurve will return an "Error" when it doesn't find a match for a target
# These are not real errors and can be safely ignored.
```

### 2.6 MAST: Hubble Catalog of Variables ([HCV](https://archive.stsci.edu/hlsp/hcv))
The function to retrieve lightcurves from HCV currently uses their API; based on this [example](https://archive.stsci.edu/hst/hsc/help/HCV/HCV_API_demo.html). This search is not efficient at scale and we expect it to be replaced in the future.

```{code-cell} ipython3
HCVstarttime = time.time()

HCV_radius = 1.0/3600.0 # radius = 1 arcsec
# get HCV light curves
df_lc_HCV = HCV_get_lightcurves(sample_table, HCV_radius)

# add the resulting dataframe to all other archives
df_lc.append(df_lc_HCV)

print('HCV search took:', time.time() - HCVstarttime, 's')
```

## 3. Find light curves for these targets in relevant, non-NASA catalogs

+++

### 3.1 Gaia
The function to retrieve Gaia light curves accesses the Gaia DR3 "source lite" catalog using an astroquery search with a table upload to do the join with the Gaia phtometry. This is currently the fastest way to access light curves from Gaia at scale.

```{code-cell} ipython3
gaiastarttime = time.time()

# get Gaia light curves
df_lc_gaia = Gaia_get_lightcurve(sample_table, 1/3600., 0)

# add the resulting dataframe to all other archives
df_lc.append(df_lc_gaia)

print('gaia search took:', time.time() - gaiastarttime, 's')
```

### 3.3 Icecube neutrinos

There are several [catalogs](https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018) (basically one for each year of IceCube data from 2008 - 2018). The following code creates a large catalog by combining
all the yearly catalogs.
The IceCube catalog contains Neutrino detections with associated energy and time and approximate direction (which is uncertain by half-degree scales....). Usually, for active events only one or two Neutrinos are detected, which makes matching quite different compared to "photons". For our purpose, we will list the top 3 events in energy that are within a given distance to the target.

This time series (time vs. neutrino energy) information is similar to photometry. We choose to storing time and energy in our data structure, leaving error = 0. What is __not__ stored in this format is the distance or angular uncertainty of the event direction.

```{code-cell} ipython3
icecubestarttime = time.time()

# get icecube datapoints
df_lc_icecube = Icecube_get_lightcurve(sample_table ,
                                   icecube_select_topN = 3)

# add the resulting dataframe to all other archives
df_lc.append(df_lc_icecube)

print('icecube search took:', time.time() - icecubestarttime, 's')
end_serial = time.time()
```

```{code-cell} ipython3
# benchmarking
print('total time for serial archive calls is ', end_serial - start_serial, 's')
```

## 4. Parallel processing the archive calls

```{code-cell} ipython3
# define some variables in case the above serial cells are not run
max_fermi_error_radius = str(1.0)  
max_sax_error_radius = str(3.0)
heasarc_catalogs = {"FERMIGTRIG": max_fermi_error_radius, "SAXGRBMGRB": max_sax_error_radius}
bandlist = ["W1", "W2"]
wise_radius = 1.0 * u.arcsec
panstarrs_radius = 1.0 / 3600.0  # search radius = 1 arcsec
lk_radius = 1.0  # arcseconds
hcv_radius = 1.0 / 3600.0  # radius = 1 arcsec
```

```{code-cell} ipython3
# number of workers to use in the parallel processing pool
# this should equal the total number of archives called
n_workers = 8
```

```{code-cell} ipython3
# the ZTF call can be parallelized internally, separate from the pool launched below.
# these parallelizations are mutually exclusive, so we must turn off the internal parallelization.
ztf_nworkers = None

# note that the ZTF call is relatively slow compared to other archives.
# if you want to query for a large number of objects, it will be faster to call ZTF individually
# (code above) and use the internal parallelization. try 8-12 workers.
```

```{code-cell} ipython3
parallel_starttime = time.time()

# start a multiprocessing pool and run all the archive queries
parallel_df_lc = MultiIndexDFObject()  # to collect the results
callback = parallel_df_lc.append  # will be called once on the result returned by each archive
with mp.Pool(processes=n_workers) as pool:

    # start the processes that call the archives
    pool.apply_async(
        Gaia_get_lightcurve, (sample_table, 1/3600., 0), callback=callback
    )
    pool.apply_async(
        HEASARC_get_lightcurves, (sample_table, heasarc_catalogs), callback=callback
    )
    pool.apply_async(
        HCV_get_lightcurves, (sample_table, hcv_radius), callback=callback
    )
    pool.apply_async(
        Icecube_get_lightcurve, (sample_table , 3), callback=callback
    )
    pool.apply_async(
        Panstarrs_get_lightcurves, (sample_table, panstarrs_radius), callback=callback
    )
    pool.apply_async(
        TESS_Kepler_get_lightcurves, (sample_table, lk_radius), callback=callback
    )
    pool.apply_async(
        WISE_get_lightcurves, (sample_table,  wise_radius, bandlist), callback=callback
    )
    pool.apply_async(
        ZTF_get_lightcurve, (sample_table, ztf_nworkers), callback=callback
    )

    pool.close()  # signal that no more jobs will be submitted to the pool
    pool.join()  # wait for all jobs to complete, including the callback

parallel_endtime = time.time()

# LightKurve will return an "Error" when it doesn't find a match for a target
# These are not real errors and can be safely ignored.
```

```{code-cell} ipython3
# How long did parallel processing take?
# and look at the results
print('parallel processing took', parallel_endtime - parallel_starttime, 's')
parallel_df_lc.data
```

```{code-cell} ipython3
# Save the data for future use with ML notebook
#parquet_savename = 'output/df_lc_090723_yang.parquet'
#parallel_df_lc.data.to_parquet(parquet_savename)
#print("file saved!")
```

```{code-cell} ipython3
# Could load a previously saved file in order to plot
#parquet_loadname = 'output/df_lc_090723_yang.parquet'
#parallel_df_lc = MultiIndexDFObject()
#parallel_df_lc.data = pd.read_parquet(parquet_loadname)
#print("file loaded!")
```

## 5. Make plots of luminosity as a function of time
These plots are modelled after [van Velzen et al., 2021](https://arxiv.org/pdf/2111.09391.pdf). We show flux in mJy as a function of time for all available bands for each object. `show_nbr_figures` controls how many plots are actually generated and returned to the screen.  If you choose to save the plots with `save_ouptut`, they will be put in the output directory and labelled by sample number.

__Note__ that in the following, we can either plot the results from `df_lc` (from the serial call) or `parallel_df_lc` (from the parallel call). By default (see next cell) the output of the parallel call is used.

```{code-cell} ipython3
_ = create_figures(df_lc = parallel_df_lc, # either df_lc (serial call) or parallel_df_lc (parallel call)
                   show_nbr_figures = 5,  # how many plots do you actually want to see?
                   save_output = True ,  # should the resulting plots be saved?
                  )
```

## References

This work made use of:

&bull; Astroquery; Ginsburg et al., 2019, 2019AJ....157...98G  
&bull; Astropy; Astropy Collaboration 2022, Astropy Collaboration 2018, Astropy Collaboration 2013,    2022ApJ...935..167A, 2018AJ....156..123A, 2013A&A...558A..33A  
&bull; Lightkurve; Lightkurve Collaboration 2018, 2018ascl.soft12013L  
&bull; acstools; https://zenodo.org/record/7406933#.ZBH1HS-B0eY  
&bull; unWISE light curves; Meisner et al., 2023, 2023AJ....165...36M

```{code-cell} ipython3

```
