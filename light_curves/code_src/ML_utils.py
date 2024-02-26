import numpy as np
from scipy import stats
from scipy import interpolate
import matplotlib.pyplot as plt
import numba
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic



from tqdm import tqdm

def translate_bitwise_sum_to_labels(bitwise_sum):
    """
    Translate a bitwise sum back to the labels which were set to 1.

    Parameters:
    - bitwise_sum: Integer, the bitwise sum representing the combination of labels.
    - labels: List of strings, the labels corresponding to each bit position.

    Returns:
    - List of strings, the labels that are set to 1.
    """
    # Initialize agnlabels
    agnlabels = ['SDSS_QSO', 'WISE_Variable','Optical_Variable','Galex_Variable',
                 'Turn-on', 'Turn-off',
                 'SPIDER','SPIDER_AGN','SPIDER_BL','SPIDER_QSOBL','SPIDER_AGNBL', 
                 'TDE']
    active_labels = []
    for i, label in enumerate(agnlabels):
        # Check if the ith bit is set to 1
        if bitwise_sum & (1 << i):
            active_labels.append(label)
    return active_labels

def update_bitsums(df, label_num=64):
    '''To update the bitwise summed labels by subtracting the 64s added in'''
    
    # Extract index as a list of tuples if MultiIndex, or adjust accordingly
    index_list = list(df.index)
    
    # Prepare a new list for the updated index
    updated_index = []
    
    # Track whether any changes are made to avoid unnecessary DataFrame recreation
    changes_made = False
    
    for idx in index_list:
        current_label = int(idx[1])  # Assuming 'label' is the second level in the multi-index
        
        # Check if 64 is part of the bitwise sum
        if current_label & label_num != 0:
            new_label = current_label ^ label_num  # Calculate the new label by removing 64 using XOR
            new_idx = list(idx)
            new_idx[1] = new_label  # Update the label in the index tuple
            updated_index.append(tuple(new_idx))
            changes_made = True
        else:
            updated_index.append(idx)
    
    # If changes were made, update the DataFrame index
    if changes_made:
        df_updated = df.set_index(pd.MultiIndex.from_tuples(updated_index, names=df.index.names))
    else:
        df_updated = df  # No changes, return original DataFrame
    
    return df_updated  
    
def autopct_format(values):

    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        #return '{:.1f}%\n({v:d})'.format(pct, v=val)
        return val#'{:.1f}%'.format(pct)
    

    return my_format


def unify_lc(df_lc, redshifts, bands_inlc=['zr', 'zi', 'zg'], xres=320, numplots=1, low_limit_size=5):
    '''
    Function to preprocess and unify time dimension of light curve data with linear interpolation.

    Parameters:
    - df_lc: DataFrame with light curve data.
    - bands_inlc: List of bands to include in the analysis (default: ['zr', 'zi', 'zg']).
    - xres: Resolution for interpolation (default: 160).
    - numplots: Number of plots to display (default: 1).
    - low_limit_size: Minimum number of data points required in a band (default: 5).
    '''

    # Creating linearly spaced arrays for interpolation for different instruments
    x_ztf = np.linspace(0, 1600, xres)  # For ZTF
    x_wise = np.linspace(0, 4000, xres)  # For WISE

    # Extract unique object IDs from the DataFrame
    objids = df_lc.index.get_level_values('objectid')[:].unique()

    # Initialize variables for storing results
    printcounter = 0
    objects, dobjects, flabels, keeps, zlist = [], [], [], [], []
    colors = ["#3F51B5","#40826D","#E30022","k","orange"]

    # Iterate over each object ID
    for keepindex, obj in tqdm(enumerate(objids)):
        redshift = redshifts[obj]
        singleobj = df_lc.loc[obj, :, :, :]  # Extract data for the single object
        label = singleobj.index.unique('label')  # Get the label of the object
        bands = singleobj.loc[label[0], :, :].index.get_level_values('band')[:].unique()  # Extract bands

        keepobj = 0  # Flag to determine if the object should be kept
        # Check if the object has all required bands
        if len(np.intersect1d(bands, bands_inlc)) == len(bands_inlc):
            if printcounter < numplots:
                fig, ax = plt.subplots(figsize=(15, 4))  # Set up plot if within numplots limit

            # Initialize arrays for new interpolated Y and dY values
            obj_newy = [[] for _ in range(len(bands_inlc))]
            obj_newdy = [[] for _ in range(len(bands_inlc))]

            keepobj = 1  # Set keepobj to 1 (true) initially
            # Process each band in the included bands
            for l, band in enumerate(bands_inlc):
                band_lc = singleobj.loc[label[0], band, :]  # Extract light curve data for the band
                band_lc_clean = band_lc[band_lc.index.get_level_values('time') < 65000]
                x, y, dy = np.array(band_lc_clean.index.get_level_values('time') - band_lc_clean.index.get_level_values('time')[0]), np.array(band_lc_clean.flux), np.array(band_lc_clean.err)

                # Sort data based on time
                x2, y2, dy2 = x[np.argsort(x)], y[np.argsort(x)], dy[np.argsort(x)]

                # Check if there are enough points for interpolation
                if (len(x2) > low_limit_size) and not np.isnan(y2).any():
                    # Handle time overlaps in light curves
                    n = np.sum(x2 == 0)
                    for b in range(1, n):
                        x2[::b + 1] = x2[::b + 1] + 1 * 0.001

                    # Interpolate the data
                    f = interpolate.interp1d(x2, y2, kind='previous', fill_value="extrapolate")
                    df = interpolate.interp1d(x2, dy2, kind='previous', fill_value="extrapolate")

                    # Plot data if within the numplots limit
                    if printcounter < numplots:
                        if band in ['W1', 'W2']:
                            gline, = plt.plot(x_wise, f(x_wise), '--', label='nearest interpolation ' + str(band),color = colors[l])
                            gcolor=gline.get_color()
                        else:
                            gline, = plt.plot(x_ztf, f(x_ztf), '--', label='nearest interpolation ' + str(band),color = colors[l])
                            gcolor=gline.get_color()

                        plt.errorbar(x2, y2, dy2, capsize=1.0, marker='.', linestyle='',alpha=0.4,color=gcolor)

                    # Assign interpolated values based on the band
                    if band =='W1' or band=='W2':
                        obj_newy[l] = f(x_wise)#/f(x_wise).max()
                        obj_newdy[l] = df(x_wise)
                    else:
                        obj_newy[l] = f(x_ztf)#/f(x_ztf).max()
                        obj_newdy[l] = df(x_ztf)#/f(x_ztf).max()

                else: #don't keep objects which have less than x datapoints in any keeping bands
                    keepobj = 0

            if printcounter<numplots:
                plt.xlabel(r'$\rm Time(MJD)$',size=15)
                plt.ylabel(r'$\rm Flux(mJy)$',size=15)
                plt.legend()
                #plt.show()
                plt.savefig('output/interp_ln_lc'+str(printcounter)+'.png')
                printcounter+=1


        if keepobj and not np.isnan(obj_newy).any():
            objects.append(obj_newy)
            dobjects.append(obj_newdy)
            flabels.append(label[0])
            keeps.append(keepindex)
            zlist.append(redshift)
    return np.array(objects),np.array(dobjects),flabels,keeps,np.array(zlist)


def unify_lc_gp(df_lc,redshifts,bands_inlc=['zr','zi','zg'],xres=320,numplots=1,low_limit_size=5):
    '''
    Function to preprocess and unify the time dimension of light curve data using Gaussian Processes.

    Parameters:
    - df_lc: DataFrame with light curve data.
    - bands_inlc: List of bands to include in the analysis (default: ['zr', 'zi', 'zg']).
    - xres: Resolution for interpolation (default: 160).
    - numplots: Number of plots to display (default: 1).
    - low_limit_size: Minimum number of data points required in a band (default: 5).
    '''
    x_ztf = np.linspace(0,1600,xres).reshape(-1, 1) # X array for interpolation
    x_wise = np.linspace(0,4000,xres).reshape(-1, 1) # X array for interpolation
    objids = df_lc.index.get_level_values('objectid')[:].unique()

    #kernel = 1 * RBF(length_scale=200)
    #kernel = 1.0 * Matern(length_scale=20.0, nu=10)
    kernel = RationalQuadratic(length_scale=1, alpha=0.1)
    
    printcounter = 0
    objects,dobjects,flabels,keeps,zlist = [],[],[],[],[]
    colors = ["#3F51B5","#40826D","#E30022","k","orange"]

    for keepindex,obj in tqdm(enumerate(objids)):
        redshift = redshifts[obj]

        singleobj = df_lc.loc[obj,:,:,:]
        label = singleobj.index.unique('label')
        bands = singleobj.loc[label[0],:,:].index.get_level_values('band')[:].unique()
        keepobj = 0
        
        if len(np.intersect1d(bands,bands_inlc))==len(bands_inlc):
            if printcounter<numplots:
                fig= plt.subplots(figsize=(15,4))

            obj_newy = [ [] for _ in range(len(bands_inlc))]
            obj_newdy = [ [] for _ in range(len(bands_inlc))]

            keepobj = 1 #
            for l,band in enumerate(bands_inlc):
                band_lc = singleobj.loc[label[0], band, :]
                band_lc_clean = band_lc[band_lc.index.get_level_values('time') < 65000]
                x,y,dy = np.array(band_lc_clean.index.get_level_values('time')-band_lc_clean.index.get_level_values('time')[0]),np.array(band_lc_clean.flux),np.array(band_lc_clean.err)

                x2,y2,dy2 = x[np.argsort(x)],y[np.argsort(x)],dy[np.argsort(x)]
                if len(x2)>low_limit_size and (not np.isnan(y2).any()) and (not np.isnan(dy2).any()):
                    n = np.sum(x2==0)
                    for b in range(1,n): # this is a hack of shifting time of different lightcurves by a bit so I can interpolate!
                        x2[::b+1]=x2[::b+1]+1*0.001
                    X = x2.reshape(-1, 1)
                    gp = GaussianProcessRegressor(kernel=kernel, alpha=dy2**2)
                    gp.fit(X, y2)
                    
                    if band =='W1' or band=='W2':
                        obj_newy[l],obj_newdy[l] = gp.predict(x_wise, return_std=True)
                    else:
                        obj_newy[l],obj_newdy[l] = gp.predict(x_ztf, return_std=True)

                    if printcounter<numplots:
                        if band=='W1' or band=='W2':
                            y_pred,sigma = gp.predict(x_wise, return_std=True)
                            gpline, = plt.plot(x_wise,y_pred,'--',label='Gaussian Process Reg.'+str(band),color = colors[l])
                            gcolor= gpline.get_color()
                            plt.fill_between(x_wise.flatten(), y_pred - 1.96 * sigma,y_pred + 1.96 * sigma, alpha=0.2, color=gcolor)
                        else:
                            y_pred,sigma = gp.predict(x_ztf, return_std=True)
                            gpline, = plt.plot(x_ztf,y_pred,'--',label='Gaussian Process Reg.'+str(band),color = colors[l])
                            gcolor= gpline.get_color()
                            plt.fill_between(x_ztf.flatten(), y_pred - 1.96 * sigma,y_pred + 1.96 * sigma, alpha=0.2, color=gcolor)
                        plt.errorbar(x2,y2,dy2 , capsize = 1.0,marker='.',linestyle='',alpha=0.4,color=gcolor)

                else:
                    keepobj=0
            if (printcounter<numplots):
                #plt.title('Object '+str(obj))#+' from '+label[0]+' et al.')
                plt.xlabel(r'$\rm Time(MJD)$',size=15)
                plt.ylabel(r'$\rm Flux(mJy)$',size=15)
                plt.legend()
                #plt.show()
                plt.savefig('output/interp_gp_lc'+str(printcounter)+'.png')
                printcounter+=1
        if keepobj and not np.isnan(obj_newy).any():
            objects.append(obj_newy)
            dobjects.append(obj_newdy)
            flabels.append(label[0])
            keeps.append(keepindex)
            zlist.append(redshift)

        #if keepindex>10:
    return np.array(objects),np.array(dobjects),flabels,keeps,np.array(zlist)
            
def combine_bands(objects,bands):
    '''
    combine all lightcurves in individual bands of an object
    into one long array, by appending the indecies.
    '''
    dat = []
    for o,ob in enumerate(objects):
        obj = []
        for b in range(len(bands)):
            obj = np.append(obj,ob[b],axis=0)
        dat.append(obj)
    return np.array(dat)

def mean_fractional_variation(lc,dlc):
    '''A common way of defining variability'''
    meanf = np.mean(lc) #mean flux of all points
    varf = np.std(lc)**2
    deltaf = np.mean(dlc)**2
    if meanf<=0:
        meanf = 0.0001
    fvar = (np.sqrt(varf-deltaf))/meanf
    return fvar

def stat_bands(objects, dobjects, bands, sigmacl=5):
    '''
    Returns arrays with maximum, mean, std flux in the 5sigma clipped lightcurves of each band.
    '''
    num_bands = len(bands)
    num_objects = len(objects)
    fvar = np.zeros((num_bands, num_objects))
    maxarray = np.zeros((num_bands, num_objects))
    meanarray = np.zeros((num_bands, num_objects))

    for o, ob in enumerate(objects):
        for b in range(num_bands):
            clipped_arr, _, _ = stats.sigmaclip(ob[b], low=sigmacl, high=sigmacl)
            clipped_varr, _, _ = stats.sigmaclip(dobjects[o, b, :], low=sigmacl, high=sigmacl)

            # Check if clipped_arr is not empty
            if clipped_arr.size > 0:
                maxarray[b, o] = clipped_arr.max() if clipped_arr.size > 0 else 0
                meanarray[b, o] = clipped_arr.mean() if clipped_arr.size > 0 else 0
                fvar[b, o] = mean_fractional_variation(clipped_arr, clipped_varr) if clipped_arr.size > 0 else 0
            else:
                # Handle empty arrays by setting values to NaN or another appropriate value
                maxarray[b, o] = 0
                meanarray[b, o] = 0
                fvar[b, o] = 0

    return fvar, maxarray, meanarray


def normalize_mean_objects(data):
    '''
    normalize objects in all bands together by mean value.
    '''
    # normalize each databand
    row_sums = data.mean(axis=1)
    return data / row_sums[:, np.newaxis]

def normalize_max_objects(data):
    '''
    normalize objects in all bands together by max value.
    '''
    # normalize each databand
    row_sums = data.max(axis=1)
    return data / row_sums[:, np.newaxis]

def normalize_clipmax_objects(data,maxarr,band =1):
    '''
    normalize combined data array by by max value after clipping the outliers in one band (second band here).
    '''
    d2 = np.zeros_like(data)
    for i,d in enumerate(data):
        if band<=np.shape(maxarr)[0]:
            d2[i] = (d/maxarr[band,i])
        else:
            d2[i] = (d/maxarr[0,i])
    return d2

# Shuffle before feeding to umap
def shuffle_datalabel(data,labels):
    """shuffles the data, labels and also returns the indecies """
    p = np.random.permutation(len(data))
    data2 = data[p,:]
    fzr=np.array(labels)[p.astype(int)]
    return data2,fzr,p

@numba.njit()
def dtw_distance(series1, series2):
    """
    Returns the DTW similarity distance between two 2-D
    timeseries numpy arrays.
    Arguments:
        series1, series2 : array of shape [n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared
    Returns:
        DTW distance between sequence 1 and 2
    """
    l1 = series1.shape[0]
    l2 = series2.shape[0]
    E = np.empty((l1, l2))

    # Fill First Cell
    E[0][0] = np.square(series1[0] - series2[0])

    # Fill First Column
    for i in range(1, l1):
        E[i][0] = E[i - 1][0] + np.square(series1[i] - series2[0])

    # Fill First Row
    for i in range(1, l2):
        E[0][i] = E[0][i - 1] + np.square(series1[0] - series2[i])

    for i in range(1, l1):
        for j in range(1, l2):
            v = np.square(series1[i] - series2[j])

            v1 = E[i - 1][j]
            v2 = E[i - 1][j - 1]
            v3 = E[i][j - 1]

            if v1 <= v2 and v1 <= v3:
                E[i][j] = v1 + v
            elif v2 <= v1 and v2 <= v3:
                E[i][j] = v2 + v
            else:
                E[i][j] = v3 + v

    return np.sqrt(E[-1][-1])

def stretch_small_values_arctan(data, factor=1.0):
    """
    Stretch small values in an array using the arctan function.

    Parameters:
    - data (numpy.ndarray): The input array.
    - factor (float): A factor to control the stretching. Larger values will stretch more.

    Returns:
    - numpy.ndarray: The stretched array.
    """
    stretched_data = np.arctan(data * factor)
    return stretched_data
