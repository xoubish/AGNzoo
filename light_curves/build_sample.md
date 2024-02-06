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
By Shooby, Last edit Feb 6th

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
from astroquery.ipac.ned import Ned

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
agnlabels = {'SDSS_QSO', 'WISE_R90', 'WISE_Variable','Galex_Variable','Optical_Variable',
             'Palomar_Variable', 'Turn-on', 'Turn-off',
             'SPIDER', 'SPIDER_Broadline', 'SPIDER_QSO', 'SPIDER_AGN',
             'TDE','Fermi_blazar'}

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
print(len(df))
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
print('objects in df now',len(df))

uspideragn_labels = ['SPIDER_AGN' for ra in a['SDSS_RA'][uspideragn]]
update_or_append_multiple(a['SDSS_RA'][uspideragn],a['SDSS_DEC'][uspideragn],a['Z_BEST'][uspideragn],uspideragn_labels)
print('objects in df now',len(df))

```

# Variable AGNs (WISE, COSMOS VLT, Palomar, Galex, )

```python
VAGN = pd.read_csv('data/WISE_MIR_variable_AGN_with_PS1_photometry_and_SDSS_redshift.csv')
uwise = (VAGN['SDSS_redshift']>0)&(VAGN['SDSS_redshift']<zmax)
vagn_labels = ['WISE_Variable' for ra in VAGN['SDSS_RA'][uwise]]
print(len(vagn_labels))
update_or_append_multiple(VAGN['SDSS_RA'][uwise],VAGN['SDSS_Dec'][uwise],VAGN['SDSS_redshift'][uwise],vagn_labels)

```

```python
paper = Ned.query_refcode('2019A&A...627A..33D') #optically variable AGN in cosmos
up = (paper['Redshift']>0)&(paper['Redshift']<=zmax)
paper_labels = ['Optical_Variable' for ra in paper['RA'][up]]
update_or_append_multiple(paper['RA'][up], paper['DEC'][up],paper['Redshift'][up],paper_labels)
print(len(paper_labels))
```

```python
paper = Ned.query_refcode('2022ApJ...933...37W') #Galex Variable
up = (paper['Redshift']>0)&(paper['Redshift']<=zmax)
paper_labels = ['Galex_Variable' for ra in paper['RA'][up]]
update_or_append_multiple(paper['RA'][up], paper['DEC'][up],paper['Redshift'][up],paper_labels)
print(len(paper_labels))
```

```python
paper = Ned.query_refcode('2020ApJ...896...10B') #Palomar Variable
up = (paper['Redshift']>0)&(paper['Redshift']<=zmax)
paper_labels = ['Palomar_Variable' for ra in paper['RA'][up]]
update_or_append_multiple(paper['RA'][up], paper['DEC'][up],paper['Redshift'][up],paper_labels)
print(len(paper_labels))
```

# Fermi (gamma ray) Blazars

<!-- #raw -->
paper = Ned.query_refcode('2015ApJ...810...14A') #FERMI BLAZERS
up = (paper['Redshift']>0)&(paper['Redshift']<=zmax)
paper_labels = ['Fermi_blazar' for ra in paper['RA'][up]]
update_or_append_multiple(paper['RA'][up], paper['DEC'][up],paper['Redshift'][up],paper_labels)
print(len(paper_labels))
<!-- #endraw -->

# TDEs

```python
data = """
1 ZTF18acaqdaa AT2018iih 262.0163662 30.6920758 0.212 van Velzen et al. (2021) TDE-He
2 ZTF18acnbpmd AT2018jbv 197.6898587 8.5678292 0.340 Hammerstein et al. (2023) TDE-featureless
3 ZTF19aabbnzo AT2018lna 105.8276892 23.0290953 0.0914 van Velzen et al. (2021) TDE-H+He
4 ZTF19aaciohh AT2019baf 268.0005082 65.6266546 0.0890 This paper; J. Somalwar et al. (2023, in preparation) Unknown
5 ZTF17aaazdba AT2019azh 123.3206388 22.6483180 0.0222 Hinkle et al. (2021) TDE-H+He
6 ZTF19aakswrb AT2019bhf 227.3165243 16.2395720 0.121 van Velzen et al. (2021) TDE-H
7 ZTF19aaniqrr AT2019cmw 282.1644974 51.0135422 0.519 This paper; J. Wise et al. (2023, in preparation) TDE-featureless
8 ZTF19aapreis AT2019dsg 314.2623552 14.2044787 0.0512 Stein et al. (2021) TDE-H+He
9 ZTF19aarioci AT2019ehz 212.4245268 55.4911223 0.0740 van Velzen et al. (2021) TDE-H
10 ZTF19abzrhgq AT2019qiz 71.6578313 −10.2263602 0.0151 Nicholl et al. (2020) TDE-H+He
11 ZTF19acspeuw AT2019vcb 189.7348778 33.1658869 0.0890 Hammerstein et al. (2023) TDE-H+He
12 ZTF20aabqihu AT2020pj 232.8956925 33.0948917 0.0680 Hammerstein et al. (2023) TDE-H+He
13 ZTF20abfcszi AT2020mot 7.8063109 85.0088329 0.0690 Hammerstein et al. (2023) TDE-H+He
14 ZTF20abgwfek AT2020neh 230.3336852 14.0696032 0.0620 Angus et al. (2022) TDE-H+He
15 ZTF20abnorit AT2020ysg 171.3584535 27.4406021 0.277 Hammerstein et al. (2023) TDE-featureless
16 ZTF20acaazkt AT2020vdq 152.2227354 42.7167535 0.0450 This paper; J. Somalwar et al. (2023, in preparation) Unknown
17 ZTF20achpcvt AT2020vwl 232.6575481 26.9824432 0.0325 Hammerstein et al. (2021a) TDE-H+He
18 ZTF20acitpfz AT2020wey 136.3578499 61.8025699 0.0274 Arcavi et al. (2020) TDE-H+He
19 ZTF20acnznms AT2020yue 165.0013942 21.1127532 0.204 This paper TDE-H?
20 ZTF20acvezvs AT2020abri 202.3219785 19.6710235 0.178 This paper Unknown
21 ZTF20acwytxn AT2020acka 238.7581288 16.3045292 0.338 Hammerstein et al. (2021b) TDE-featureless
22 ZTF21aaaokyp AT2021axu 176.6514953 30.0854257 0.192 Hammerstein et al. (2021c) TDE-H+He
23 ZTF21aakfqwq AT2021crk 176.2789219 18.5403839 0.155 This paper TDE-H+He?
24 ZTF21aanxhjv AT2021ehb 46.9492531 40.3113468 0.0180 Yao et al. (2022a) TDE-featureless
25 ZTF21aauuybx AT2021jjm 219.8777384 −27.8584845 0.153 Yao et al. (2021d) TDE-H
26 ZTF21abaxaqq AT2021mhg 4.9287185 29.3168745 0.0730 Chu et al. (2021b) TDE-H+He
27 ZTF21abcgnqn AT2021nwa 238.4636684 55.5887978 0.0470 Yao et al. (2021b) TDE-H+He
28 ZTF21abhrchb AT2021qth 302.9121723 −21.1602187 0.0805 This paper TDE-coronal
29 ZTF21abjrysr AT2021sdu 17.8496154 50.5749060 0.0590 Chu et al. (2021c) TDE-H+He
30 ZTF21abqhkjd AT2021uqv 8.1661654 22.5489257 0.106 Yao (2021) TDE-H+He
31 ZTF21abqtckk AT2021utq 229.6212498 73.3587323 0.127 This paper TDE-H
32 ZTF21abxngcz AT2021yzv 105.2774821 40.8251799 0.286 Chu et al. (2022) TDE-featureless
33 ZTF21acafvhf AT2021yte 103.7697396 12.6341503 0.0530 Yao et al. (2021a) TDE-H+He
"""

# Split the data into lines
lines = data.strip().split('\n')

# Loop through each line, extracting the required information
ras,decs,redshifts=[],[],[]
for line in lines:
    parts = line.split()
    id_ = parts[0]
    ra = float(parts[3].replace('−', '-'))
    dec = float(parts[4].replace('−', '-'))
    redshift = float(parts[5])
    if (redshift >0)&(redshift<zmax): 
        ras.append(ra)
        decs.append(dec)
        redshifts.append(redshift)

ras,decs,redshifts = np.array(ras),np.array(decs),np.array(redshifts)
TDE_labels = ['TDE' for ra in ras]

update_or_append_multiple(ras, decs,redshifts,TDE_labels)
print(len(TDE_labels))
```

# Changing Looks (dividing to on and off)

```python


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


```

# WISE R90

```python
# 
r90 = fits.getdata('data/J_ApJS_234_23_r90cat.dat.gz.fits.gz',1)
r90.columns
```

```python

```
