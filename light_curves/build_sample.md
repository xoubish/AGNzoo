---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Build AGN final catalog sample 
Last edit Feb5th

```python
import sys
sys.path.append('code_src/')

import os
import time
import astropy.units as u
import pandas as pd
import numpy as np
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from data_structures import MultiIndexDFObject

from astroquery.sdss import SDSS

from sample_selection import (clean_sample, get_green_sample, get_hon_sample, get_lamassa_sample,
                              get_lopeznavas_sample, get_lyu_sample, get_macleod16_sample, get_macleod19_sample, get_paper_sample,
                              get_ruan_sample, get_SDSS_sample, get_sheng_sample, get_yang_sample, nonunique_sample)
import logging

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
```

```python
# Initialize agnlabels
agnlabels = {'SDSS_QSO', 'WISE_R90', 'WISE_Variable', 'Turn-on', 'Turn-off',
             'SPIDER', 'SPIDER_Broadline', 'SPIDER_QSO', 'SPIDER_AGN',
             'TDE'}

# Create an empty pandas DataFrame
columns = ['SkyCoord', 'redshift'] + list(agnlabels)
df = pd.DataFrame(columns=columns)
# Initialize label columns to 0
for label in agnlabels:
    df[label] = 0
# Function to check if a coordinate is close to any existing ones
def is_close(new_coord, threshold_arcsec=1):
    global df
    if df.empty:
        return False, None
    existing_coords = SkyCoord(df['SkyCoord'].tolist())
    sep = new_coord.separation(existing_coords)
    close_idx = sep < threshold_arcsec * u.arcsec
    if close_idx.any():
        return True, df[close_idx].index[0]
    else:
        return False, None

def update_or_append_multiple(ras, decs, redshifts, labels):
    global df  # Make sure df is recognized as the global DataFrame
    new_rows = []  # Prepare a list to collect new rows
    for ra, dec, redshift, label in zip(ras, decs, redshifts, labels):
        new_coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
        exists, idx = is_close(new_coord)

        if exists:
            # Update existing row
            df.at[idx, label] = 1
            df.at[idx, 'redshift'] = redshift
        else:
            # Prepare a new row as a DataFrame instead of Series for consistency with pd.concat
            new_row = pd.DataFrame([{**{'SkyCoord': new_coord, 'redshift': redshift}, **{l: 1 if l == label else 0 for l in agnlabels}}])
            new_rows.append(new_row)

    # Append all new rows at once using pd.concat if there are any
    if new_rows:
        df = pd.concat([df, *new_rows], ignore_index=True)
```

## Add SDSS QSO from DR16

```python
num = 100
query = "SELECT TOP " + str(num) + " specObjID, ra, dec, z FROM SpecObj \
WHERE ( z > 0.1 AND z < 1.0 AND class='QSO' AND zWARNING=0 )"
if num>0:
    res = SDSS.query_sql(query, data_release = 16)
    for r in res:
        update_or_append_multiple([r['ra']], [r['dec']], [r['z']], ['SDSS_QSO'])
df
```

# SPIDERS 
SPectroscopic IDentfication of ERosita Sources- (value added SDSS DR16 which are all detected in xray) https://www.sdss.org/dr18/bhm/programs/spiders/

```python
a = fits.getdata('data/VAC_SPIDERS_2RXS_DR16.fits',1)
#print(a.columns)
print(np.unique(a['DR16_CLASS']))
#print(np.unique(a['DR16_SUBCLASS']))
print('CLASS GALAXY DR16 SPIDER:',len(a['DR16_SUBCLASS'][a['DR16_CLASS']=='GALAXY']),' subclasses: ',np.unique(a['DR16_SUBCLASS'][a['DR16_CLASS']=='GALAXY']))
print('CLASS QSO DR16 SPIDER:',len(a['DR16_SUBCLASS'][a['DR16_CLASS']=='QSO']),' subclasses: ',np.unique(a['DR16_SUBCLASS'][a['DR16_CLASS']=='QSO']))

```

```python
zmax = 1.0
# all spider that can be AGN/QSO
uspider = (a['DR16_CLASS']!='STAR')&(a['DR16_SUBCLASS']!='STARFORMING')&(a['DR16_SUBCLASS']!='STARFORMING BROADLINE')&(a['DR16_SUBCLASS']!='STARBURST')&(a['DR16_SUBCLASS']!='STARBURST BROADLINE')&(a['Z_BEST']>0)&(a['Z_BEST']<zmax)

# all GALAXY and QSO AGN Broadlines
uspiderbroad = (a['DR16_CLASS']!='STAR')&(a['DR16_SUBCLASS']=='AGN BROADLINE')&(a['Z_BEST']<=zmax)&(a['Z_BEST']>=0.0)
uspideragn = (a['DR16_CLASS']!='STAR')&(a['DR16_SUBCLASS']=='AGN')&(a['Z_BEST']<=zmax)&(a['Z_BEST']>=0.0)

# Gal AGNs
ugal_agn = (a['DR16_CLASS']=='GALAXY') &(a['DR16_SUBCLASS']=='AGN')&(a['Z_BEST']<=zmax)&(a['Z_BEST']>=0.0)
ugal_agnbl = (a['DR16_CLASS']=='GALAXY') &(a['DR16_SUBCLASS']=='AGN BROADLINE')&(a['Z_BEST']<=zmax)&(a['Z_BEST']>=0.0)
uqso_agn = (a['DR16_CLASS']=='QSO') &(a['DR16_SUBCLASS']=='AGN')&(a['Z_BEST']<=zmax)&(a['Z_BEST']>=0.0)
uqso_agnbl = (a['DR16_CLASS']=='QSO') &(a['DR16_SUBCLASS']=='AGN BROADLINE')&(a['Z_BEST']<=zmax)&(a['Z_BEST']>=0.0)

print(len(a[uspider]),len(a[uspiderbroad]),len(a[uspideragn]),len(a[ugal_agn]),len(a[ugal_agnbl]),len(a[uqso_agn]),len(a[uqso_agnbl]))

o = plt.hist(a['Z_BEST'][ugal_agn],histtype='step',label='Galaxy AGNs')
o = plt.hist(a['Z_BEST'][ugal_agnbl],histtype='step',label = 'Galaxy AGN_BL')
o = plt.hist(a['Z_BEST'][uqso_agn],histtype='step',label='QSO AGN')
o = plt.hist(a['Z_BEST'][uqso_agnbl],histtype='step',label='QSO AGN_BL')

o = plt.hist(a['Z_BEST'][uspiderbroad],histtype='step',label='All AGN broad')
o = plt.hist(a['Z_BEST'][uspideragn],histtype='step',label='All AGN non-broad')


plt.legend()
plt.xlabel('redshift')
plt.ylabel('counts')
```

```python
#long
#uspider_labels = ['SPIDER' for ra in a['SDSS_RA'][uspider]]
#update_or_append_multiple(a['SDSS_RA'][uspider],a['SDSS_DEC'][uspider],a['Z_BEST'][uspider],uspider_labels)

uspiderbl_labels = ['SPIDER_BLAGN' for ra in a['SDSS_RA'][uspiderbroad]]
update_or_append_multiple(a['SDSS_RA'][uspiderbroad],a['SDSS_DEC'][uspiderbroad],a['Z_BEST'][uspiderbroad],uspiderbl_labels)

uspideragn_labels = ['SPIDER_AGN' for ra in a['SDSS_RA'][uspideragn]]
update_or_append_multiple(a['SDSS_RA'][uspideragn],a['SDSS_DEC'][uspideragn],a['Z_BEST'][uspideragn],uspideragn_labels)
df

```

# WISE R90

```python
# 
r90 = fits.getdata('data/J_ApJS_234_23_r90cat.dat.gz.fits.gz',1)
r90.columns
```

```python

```

```python
def build_sample():
    '''Putting together a sample of SDSS quasars, WISE variable AGNs,
    TDEs, Changing look AGNs, .. coordinates from different
    papers.'''

    coords = []
    labels = []

    get_lamassa_sample(coords, labels)  #2015ApJ...800..144L
    get_macleod16_sample(coords, labels) #2016MNRAS.457..389M
    get_ruan_sample(coords, labels) #2016ApJ...826..188R
    get_macleod19_sample(coords, labels)  #2019ApJ...874....8M
    get_sheng_sample(coords, labels)  #2020ApJ...889...46S
    get_green_sample(coords, labels)  #2022ApJ...933..180G
    get_lyu_sample(coords, labels)  #z32022ApJ...927..227L
    get_lopeznavas_sample(coords, labels)  #2022MNRAS.513L..57L
    get_hon_sample(coords, labels)  #2022MNRAS.511...54H
    get_yang_sample(coords, labels)   #2018ApJ...862..109Y

    # Variable AGN sample from Ranga/Andreas:
    VAGN = pd.read_csv('../data/WISE_MIR_variable_AGN_with_PS1_photometry_and_SDSS_redshift.csv')
    vagn_coords = [SkyCoord(ra, dec, frame='icrs', unit='deg') for ra, dec in zip(VAGN['SDSS_RA'], VAGN['SDSS_Dec'])]
    vagn_labels = ['WISE-Variable' for ra in VAGN['SDSS_RA']]
    coords.extend(vagn_coords)
    labels.extend(vagn_labels)

    #now get some "normal" QSOs for use in the classifier
    #there are ~500K of these, so choose the number based on
    #a balance between speed of running the light curves and whatever
    #the ML algorithms would like to have
    num_normal_QSO = 2000
    get_SDSS_sample(coords, labels, num_normal_QSO)

    ## ADD TDEs to the sample, manually copied the TDE ZTF names from Hammerstein et al. 2023
    #tde_names = ['ZTF18aabtxvd','ZTF18aahqkbt','ZTF18abxftqm','ZTF18acaqdaa','ZTF18acpdvos','ZTF18actaqdw','ZTF19aabbnzo','ZTF18acnbpmd','ZTF19aakiwze','ZTF19aakswrb','ZTF17aaazdba','ZTF19aapreis','ZTF19aarioci','ZTF19abhejal','ZTF19abhhjcc','ZTF19abidbya','ZTF19abzrhgq','ZTF19accmaxo','ZTF20aabqihu','ZTF19acspeuw','ZTF20aamqmfk','ZTF18aakelin','ZTF20abjwvae','ZTF20abfcszi','ZTF20abefeab','ZTF20abowque','ZTF20abrnwfc','ZTF20acitpfz','ZTF20acqoiyt', 'ZTF20abnorit']
    #TDE_id2coord(tde_names,coords,labels)


    get_paper_sample('2015ApJ...810...14A','FermiBL',coords,labels)
    get_paper_sample('2019A&A...627A..33D','Cicco19',coords,labels)
    get_paper_sample('2022ApJ...933...37W','Galex variable 22',coords,labels)
    get_paper_sample('2020ApJ...896...10B','Palomar variable 20',coords,labels)

    #To remove duplicates from the list if combining multiple references clean_sample can be used
    # the call below with nonunique_sample just changes the structure to mimic the output of clean sample
    coords_list, labels_list = nonunique_sample(coords, labels)
    print('final sample: ',len(coords))
    return coords_list,labels_list


```
