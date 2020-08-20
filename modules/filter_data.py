import math
import numpy as np
import read
from astropy.stats import sigma_clip
from scipy.stats import binned_statistic 
from reduced_starselection import new_flux_err as nfe
from reduced_starselection import new_flux as nf
# Creates mask array where valid entries
# are only those with value > p * max(data)
def margin(arr, p):
    maxval = np.amax(arr)
    entries = np.zeros((1,len(arr)),dtype = bool)
    for i in range(0,len(arr)):
        if arr[i] <= p*maxval: entries[0][i] = True
        else: entries[0][i] = False
    
    result = np.ma.array(arr,mask=entries[0,:])
    return result

# Returns ratio: # false entries / total number of entries (for a bool-type array).
def false_entries(arr):
    size = len(arr)
    count = 0.
    for i in range(0,size): 
        if arr[i] == False: count += 1.
    return count / float(size)

# Given several mask arrays, returns their logical product. 
def mask_prod(arr):
    length = len(arr[:,0])
    result  = arr[0,:]
    for i in range(1,length):
        result = result + arr[i,:]
    return result

# Given a mask, and photometry data, returns the corresponding masked photometry data. 
def mask_data(data_,mask_def):
    # 0 : BJD
    # 1 : AIRMASS
    # 2 : REL_FLUX_T
    # 3 : REL_FLUX_T_ERR
    # 4 : REL_FLUX_SNR_T
    # 5 : REL_FlUX_C
    # 6 : REL_FLUX_C_ERR
    # 7 : REL_FLUX_SNR_C 
    masked_bjd_ = np.ma.masked_array(data=data_[0][0,:],mask=mask_def).compressed()
    masked_airmass_ = np.ma.masked_array(data=data_[1][0,:],mask=mask_def).compressed()
   
    size = len(masked_bjd_)
    t    = len(data_[2][:,0])
    masked_bjd = np.zeros((1,size),dtype = np.float64)
    masked_airmass = np.zeros((1,size),dtype = np.float64)
    
    masked_bjd[0] = masked_bjd_
    masked_airmass[0] = masked_airmass_
    
    masked_rel_flux_t     = np.zeros((t,size),dtype=np.float64)
    masked_rel_flux_t_err = np.zeros((t,size),dtype=np.float64)
    masked_rel_flux_t_snr = np.zeros((t,size),dtype=np.float64)
    for i in range(0,t):
        masked_rel_flux_t[i,:]     = np.ma.masked_array(data=data_[2][i,:],mask=mask_def).compressed()
        masked_rel_flux_t_err[i,:] = np.ma.masked_array(data=data_[3][i,:],mask=mask_def).compressed()
        masked_rel_flux_t_snr[i,:] = np.ma.masked_array(data=data_[4][i,:],mask=mask_def).compressed()
        
    c = len(data_[5][:,0])
    masked_rel_flux_c     = np.zeros((c,size),dtype=np.float64)
    masked_rel_flux_c_err = np.zeros((c,size),dtype=np.float64)
    masked_rel_flux_c_snr = np.zeros((c,size),dtype=np.float64)
    for i in range(0,c):
        masked_rel_flux_c[i,:]     = np.ma.masked_array(data=data_[5][i,:],mask=mask_def).compressed()
        masked_rel_flux_c_err[i,:] = np.ma.masked_array(data=data_[6][i,:],mask=mask_def).compressed()
        masked_rel_flux_c_snr[i,:] = np.ma.masked_array(data=data_[7][i,:],mask=mask_def).compressed()
        
    return [masked_bjd,masked_airmass,masked_rel_flux_t,masked_rel_flux_t_err,masked_rel_flux_t_snr,masked_rel_flux_c,masked_rel_flux_c_err,masked_rel_flux_c_snr]

# Given a mask and arrays for target and comparison stars aperture peak pixel counts,
# returns their masked versions. 
def mask_peaks(peaks_,mask_def):
    t    = len(peaks_[0][:,0])
    c    = len(peaks_[1][:,0])
    size = len(np.ma.masked_array(data = peaks_[0][0,:],mask=mask_def).compressed())
    
    masked_t_peaks = np.zeros((t,size),dtype=np.float64)
    masked_c_peaks = np.zeros((c,size),dtype=np.float64)
    for i in range(0,t):
        masked_t_peaks[i,:] = np.ma.masked_array(data=peaks_[0][i,:],mask=mask_def).compressed()
    for i in range(0,c):
        masked_c_peaks[i,:] = np.ma.masked_array(data=peaks_[1][i,:],mask=mask_def).compressed()
        
    return [masked_t_peaks,masked_c_peaks]

# Computes logical masks for comp. stars rel. fluxes (or targets, if needed) corresponding
# to a sigma clipping with specified clipping factors. A variable type_ allows deciding
# if the sigma clipping is performed w.r.t. to the mean or the median of each rel. flux. 
# It can also be used to sigma-clip target relative fluxes. 

# INPUT
# rel_flux: np.array corresponding to the rel. fluxes of the comp. stars 
#           (see REL_FLUX_C, photometry, read.py module).
# type_   : bool-type var. type_ == True -> sigma clip w.r.t. mean. type == False -> sigma clip w.r.t. median. 
# su, sl  : data is clipped if it falls outside (m - stdev*sl,m + stdev*su) 
#           where m = mean or median depending on the value of type_.

# OUTPUT
# masks: np.array, dtype=bool, dimensions #{comp. stars} x #{data points}. 
#        masks[i][j] = True means data point j from comp. star i should be discarded. 
def sigma_clip_stars(rel_flux,type_,su,sl):
    c =  len(rel_flux[:,0])
    size = len(rel_flux[0,:])
    masks = np.zeros((c,size),dtype = bool)
    if type_ == True:
        for i in range(0,c):
            Ci = rel_flux[i]
            sc = sigma_clip(Ci,sigma_upper = su,sigma_lower = sl, maxiters = 4, cenfunc = np.mean, masked = True, return_bounds = False)
            masks[i] = sc.mask
    if type_ == False:
        for i in range(0,c):
            Ci = rel_flux[i]
            sc = sigma_clip(Ci,sigma_upper = su,sigma_lower = sl, maxiters = 4, cenfunc = np.median, masked = True, return_bounds = False)
            masks[i] = sc.mask
        
    return masks

# Computes logical masks for comp. stars rel. flux SNRs (or targets, if needed) corresponding
# to a numerical clipping with specified p factor (see margin function).

# INPUT
# rel_flux_snr: np.array corresponding to the rel. flux SNRs of the comp. stars
#               (see REL_FLUX_SNR_C, photometry, read.py module).
# p           : see margin function above. 

# OUTPUT
# masks: np.array, dtype=bool, dimensions #{comp. stars} x #{data points}. 
#        masks[i][j] = True means data point j from comp. star i should be discarded. 
def snr_clip_stars(rel_flux_snr,p):
    c =  len(rel_flux_snr[:,0])
    size = len(rel_flux_snr[0,:])
    masks = np.zeros((c,size),dtype = bool)
    for i in range(0,c):
        Ci = rel_flux_snr[i]
        masks[i] = margin(Ci,p).mask
        
    return masks


# Computes mask product of all masks produced by functions sigma_clip_stars, snr_clip_stars
# among other quantities. 

# INPUT
# rel_flux, rel_flux_snr, type_, su, sl, p: see sigma_clip_stars and snr_clip_stars above. 

# OUTPUT: a list with 4 elements.
# 1. final_mask                    : mask product of all masks produced by functions 
#                                    sigma_clip_stars, snr_clip_stars. 
# 2. data_ratio                    : # false entries / # total entries in final_mask 
#                                    (ratio of data, over 1, that is reused).
# 3. sigma_clip_mask, snr_clip_mask: product of masks produced by sigma_clip_stars and 
#                                                                 snr_clip_stars  , respectively.
def filter_stars(rel_flux,rel_flux_snr,type_,su,sl,p):
    sigma_clip_mask = mask_prod(sigma_clip_stars(rel_flux,type_,su,sl))
    snr_clip_mask   = mask_prod(snr_clip_stars(rel_flux_snr,p))
    
    size = len(rel_flux[0,:])
    sigma_snr = np.zeros((2,size),dtype=bool)
    sigma_snr[0] = sigma_clip_mask
    sigma_snr[1] = snr_clip_mask
    
    final_mask = mask_prod(sigma_snr)
    data_ratio = false_entries(final_mask)
    result = [final_mask,data_ratio,sigma_clip_mask,snr_clip_mask]
    return result

# Computes masks corresponding to star aperture peak counts
# for target and comparison stars, given a peak count limit, lim_.
def saturated(peaks_,lim_):
    size = len(peaks_[0][0,:])
    targets = len(peaks_[0][:,0])
    comps   = len(peaks_[1][:,0])
    
    peaks_targets = peaks_[0]
    peaks_comps   = peaks_[1]
    
    sat_targets = np.zeros((targets,size),dtype=bool)
    sat_comps   = np.zeros((comps,size),  dtype=bool)
    
    disc_config = [0 for _ in range(0,comps)]
    
    for i in range(0,size):
        for k in range(0,targets):
            if peaks_targets[k][i] >= lim_: sat_targets[k][i] = True
        for k in range(0,comps):
            if peaks_comps[k][i] >= lim_: 
                sat_comps[k][i] = True
                disc_config[k] = 1
            
    return [sat_targets,sat_comps,disc_config]
        
# Computes mask that accounts for incorrect data values
# such as relative fluxes above 1 (or below 0), or SNR values
# unreasonably high (or low).
def threshold(arr,lim_inf,lim_sup):
    size = len(arr)
    mask = np.zeros((1,size),dtype=bool)
    for i in range(0,size):
        if arr[i] >= lim_sup or arr[i] <= lim_inf: mask[0][i] = True
    
    return mask

# Computes binned fluxes with corresponding weighted errors
# for a specified array of rel. fluxes
# given a bin size and its fluxes. 
def to_bin(arrx,arry,erry,bin_size):
    ti = arrx[0]
    tf = arrx[len(arrx) - 1]
    number_of_bins = int(abs(tf - ti)/bin_size)
    result = binned_statistic(x=arrx,values=arry*erry**(-2),statistic='sum',bins=number_of_bins)
    bin_stat    = result[0]
    bin_edges   = result[1]
    bin_indices = result[2]
    w_factors   = np.zeros((1,len(bin_stat)),dtype=np.float64)[0]
    bin_midpoints = np.zeros((1,number_of_bins),dtype=np.float64)[0]
    bin_err       = np.zeros((1,number_of_bins),dtype=np.float64)[0]
    for i in range(0,number_of_bins):
        bin_midpoints[i] = .5 * (bin_edges[i] + bin_edges[i + 1])
    
    j = 0
    size = len(arry)
    in_bin = np.zeros((1,number_of_bins),dtype=np.float64)[0]
    for i in range(0,number_of_bins):
        if i <= number_of_bins - 2: 
            while bin_edges[i] <= arrx[j] and arrx[j] < bin_edges[i + 1]:
                in_bin[i] += 1
                j += 1
        else: 
            while bin_edges[i] <= arrx[j] and arrx[j] <= bin_edges[i + 1]:
                in_bin[i] += 1
                j += 1
                if j >= size: break

    for j in range(0,len(bin_stat)): 
        if bin_stat[j] == 0: bin_stat[j] = float('nan')
    for i in range(0,size):
        if bin_stat[bin_indices[i] - 1] != float('nan'):
            w_factors[bin_indices[i] - 1] += erry[i]**(-2)
    for j in range(0,len(bin_stat)):
        if bin_stat[j] != float('nan'): bin_stat[j] /= w_factors[j]
    for i in range(0,size):
        if in_bin[bin_indices[i] - 1] <= 1 or bin_stat[bin_indices[i] - 1] == float('nan'): 
            bin_err[bin_indices[i] - 1] = float('nan')
        else: 
            bin_err[bin_indices[i] - 1] += (arry[i] - bin_stat[bin_indices[i] - 1])**2 / (float(in_bin[bin_indices[i] - 1]) - 1.)
    for j in range(0,len(bin_err)): 
        if in_bin[j] != 0: 
            bin_err[j] += 1. / w_factors[j]
            bin_err[j] /= float(in_bin[j])
    bin_err = np.sqrt(bin_err)
    return [bin_stat,bin_err,bin_midpoints,bin_edges]


# Given a bin size, a certain configuration, and all
# original read data, computes the corresponding binned
# data. 
def bin_photometry(data_,bin_size,config):
    BJD = data_[0][0]
    
    FLUX = nf(data_[2],data_[5],config)
    T_FLUX = FLUX[0][0]
    C_FLUX = FLUX[1]
    
    ERR = nfe(data_[2],data_[3],data_[5],data_[6],config)
    T_ERR = ERR[0][0]
    C_ERR = ERR[1]
    
    T_BIN = to_bin(BJD,T_FLUX,T_ERR,bin_size)
    C_BIN = []
    c = len(data_[5][:,0])
    
    new_c  = []
    disc_c = []
    for i in range(0,c):
        if config[i] == 0: new_c.append(i)
        else: disc_c.append(i)
        
    discarded = len(disc_c)
    selected = len(new_c)
    for i in range(0,selected):
        C_BIN.append(to_bin(BJD,C_FLUX[i],C_ERR[i],bin_size))
    
    size = len(T_BIN[0])
    
    NEW_BJD = np.zeros((1,size))
    NEW_BJD[0] = T_BIN[2]
    
    T_NEW_FLUX = np.zeros((1,size))
    T_NEW_ERR  = np.zeros((1,size))
    T_NEW_FLUX[0] = T_BIN[0]
    T_NEW_ERR[0]  = T_BIN[1]
    
    C_NEW_FLUX = np.zeros((selected,size))
    C_NEW_ERR  = np.zeros((selected,size))
    for i in range(0,selected):
        C_NEW_FLUX[i] = C_BIN[i][0]
        C_NEW_ERR[i]  = C_BIN[i][1]
    
    t_nans     = np.zeros((1,size),dtype=bool) 
    t_err_nans = np.zeros((1,size),dtype=bool)
    t_nans[0]     = np.isnan(T_NEW_FLUX[0])
    t_err_nans[0] = np.isnan(T_NEW_ERR[0])
    
    c_nans     = np.zeros((selected,size),dtype=bool)
    c_err_nans = np.zeros((selected,size),dtype=bool)
    for i in range(0,selected): 
        c_nans[i] = np.isnan(C_NEW_FLUX[i])
        c_err_nans[i] = np.isnan(C_NEW_ERR[i])
        
    return [NEW_BJD,T_NEW_FLUX,T_NEW_ERR,C_NEW_FLUX,C_NEW_ERR,[t_nans,t_err_nans,c_nans,c_err_nans]]
    
# Computes the phase associated to the array time_ assuming
# a full cycle is completed in period_. 
def fold(time_,period_):
    n = len(time_)
    phase = np.zeros((1,n))
    t0 = time_[0]
    for i in range(1,n):
        phase[0][i]  = (time_[i] - t0) / period_
        phase[0][i] -= int(phase[0][i])
        
    return phase[0]



    
    

