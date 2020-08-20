import math
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
import sys
sys.path.append('/home/guifress/pdo/modules')
import read
import filter_data as fd
import starselection as ss
import more_stats as st

# These are simple routines for specifying
# filenames, given filtering and star selection
# optimization parameters. 
def suffix(c_filtered,c_m,c_su,c_sl,c_p,t_filtered,t_m,t_su,t_sl,t_p,var_index,arr,opts,ks,thresholds,method,k_max,N_min,peak_limit,binning,bin_interval):
    
    result = ''
    if c_filtered == False: 
        result += '_nofiltcomp'
    else:
        if c_m == True: 
            result += '_filtcomp_mean_'
            result += str(c_su) + '_' + str(c_sl) + '_' + str(c_p)
            
    if t_filtered == False: 
        result += '_nofilttarget'
    else:
        if t_m == True: 
            result += '_filttarget_mean_'
        else:
            result += '_filttarget_median_'
        result += str(t_su) + '_' + str(t_sl) + '_' + str(t_p)
        
    if var_index == True:
        result += '_varindex'
        if arr[0] == True: 
            result += '_rms'
        if arr[1] == True:
            result += '_IQR'
        if arr[2] == True:
            result += '_neumann'
        if arr[3] == True:
            result += '_chi2'
        
        result += '_o_'
        for i in range(0,len(opts)):
            if opts[i] == True:
                result += '1'
            else: 
                result += '0'
        
        result += '_k_'
        for i in range(0,len(ks)):
            result += str(ks[i])
        
        result += '_t_'
        for i in range(0,len(thresholds)):
            result += str(thresholds[i]) + '_'
    else:
        result += '_novarindex_'
        
    if method == False:
        result += 'dynamic_kmax_' + str(k_max)
    else:
        result += 'static_Nmin_' + str(N_min)
        
    result += '_pl_' + str(peak_limit)    
    
    if binning == True: 
        result += '_binned_%.2f' % (bin_interval) 
    
    return result

def file_name(prefix_,base_,suffix_):
    return prefix_ + base_ + suffix_


