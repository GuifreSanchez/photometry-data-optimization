# PREPARED FOR COMPUTATION ON RELATIVE FLUXES FROM:
#       TZ ARI.

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
import sys
sys.path.append('/home/guifress/pdo/modules')
import read
import filter_data as fd
import starselection as ss
import more_stats as st
import varindex as vi
import write
import bar
loc = ("/home/guifress/pdo/data/TZAri_TJO_R.xlsx")
weighted = True
'''
# Data organization in TOI-1266 file:
org    = [788,1,13,20]
target = [20,21,30,36,37,38,39]
comp   = [22,23,24,25,26,27,28,29,31,32,33,34,35]
ind    = [target,comp]
'''

'''
# Data organization in GJ555 file:
# org[0] = 2739 (2 epochs), org[0] = 1563 (first epoch)
org    = [1563,1,15,17]
target = [22,33]
comp   = [23,24,25,26,27,28,29,30,31,32,34,35,36,37,38]
ind    = [target,comp]
'''

# Data organization in TZAri file:
org    = [3646,1,15,17]
target = [20,33]
comp   = [21,22,23,24,25,26,27,28,29,30,31,32,34,35,36]
ind    = [target,comp]

'''
# Data organization in TOI-1266 file:
org    = [653,1,15,22]
target = [20,24,25,27,31,32,33]
comp   = [21,22,23,26,28,29,30,34,35,36,37,38,39,40,41]
ind    = [target,comp]
'''

# Specify filters that are to be used. 
threshold_filter = True
c_filter         = False
t_filter         = False
peak_filter      = True
var_index        = False
binning          = False
c_selection      = True
c_selection_snr  = False

# Sigma clip and snr numerical clip parameters for comp. star filtering:
c_su = 4.
c_sl = 4.
c_p  = 0.
c_m  = True

# Sigma clip and snr numerical clip parameters for target filtering: 
t_su = 3.
t_sl = 3.
t_p  = 0.2
t_m  = True

peak_limit = 45000.

# Options, parameters and general config. for computation of variability indices:
opts       = [True,True,True,True]
arr        = [False,False,False,False]
ks         = [1,1,1,1]
thresholds = [1.,1.,1.,0.1]
for i in range(0,4): 
    if arr[i] == True: var_index = True

# Binning parameters
comp_stars = len(comp)
bin_min  = 1. / 24.
bin_max  =  1. 
bin_interval = .5 * .25 * 1. / 24.

# Parameters and method for comp. star selection:
k_max = comp_stars - 1
N_min = 4
method = False

# Filename suffix and prefix:
# Suffix: 
suffix_ = write.suffix(c_filter,c_m,c_su,c_sl,c_p,t_filter,t_m,t_su,t_sl,t_p,var_index,arr,opts,ks,thresholds,method,k_max,N_min,peak_limit,binning,bin_interval)
# Prefix:
prefix_ = 'TZAri_'

# Reading data 
print("Reading data...")
DATA = read.photometry(loc,org,ind)
print("Done.")
print("\n")

print("Applying selected filters...")
if threshold_filter == True:
    print("\t Filtering saturated SNR and rel. flux values...")
    filter_snr      = fd.threshold(DATA[4][0],0,1000)
    filter_rel_flux = fd.threshold(DATA[2][0],0,1.)
    CDATA = fd.mask_data(DATA,filter_snr + filter_rel_flux)
    print("\t Done.")
else:
    CDATA = DATA

if c_filter == True:
    print("\t Filtering comp. star data...")
    # Filter with sigma clip and snr numerical clip all comp. stars. Combine masks:
    filter_comp = fd.filter_stars(CDATA[5],CDATA[7],c_m,c_su,c_sl,c_p)
    print("\t Done.")
else:
    filter_comp = [np.zeros((1,len(CDATA[2][0])),dtype=bool)[0]]

if t_filter == True:
    print("\t Filtering target data...")
    # Filter with sigma clip and snr numerical clip target star. Combine masks:
    filter_target = fd.filter_stars(CDATA[2],CDATA[4],t_m,t_su,t_sl,t_p)
    print("\t Done.")
else:
     filter_target = [np.zeros((1,len(CDATA[2][0])),dtype=bool)[0]]

if peak_filter == True:
    print("\t Filtering flux peak values...")
    # Filter target fluxes above peak limit. Discard comp. star with some peak flux value >= peak_limit.
    filter_peaks = fd.saturated(fd.mask_peaks(fd.read.peak_values(loc,org,ind),filter_snr + filter_rel_flux),peak_limit)
    filter_peaks_target = filter_peaks[0]
    filter_peaks_comp   = fd.mask_prod(filter_peaks[1])
    # The following line may be added if comparison stars with some aperture peak pixel count > peak_limit are to
    # be discarded.
    # peak_config         = filter_peaks[2]
    peak_config = [0 for _ in range(0,comp_stars)]
    print("\t Done.")
else:
     filter_peaks_target = [np.zeros((1,len(CDATA[2][0])),dtype=bool)[0]]
     filter_peaks_comp   =  np.zeros((1,len(CDATA[2][0])),dtype=bool)[0]
     peak_config         = [0 for _ in range(0,comp_stars)] 

# Get non-discarded data from previous filtering.
total_filter = filter_target[0] + filter_peaks_target[0] + filter_comp[0] + filter_peaks_comp
MDATA = fd.mask_data(CDATA,total_filter)

if var_index == True:
    print("\t Computing selected variability indices...")
    # Compute different var. indices for filtered data from comp. stars and
    # get configuration with discarded comp. stars due to high dispersion / high values for var. indices. 
    index_config = ss.select_combined(MDATA[5],MDATA[6],opts,ks,thresholds,arr)
    print("\t Done.")
    print("\n") 
else:
    index_config = [0 for _ in range(0,comp_stars)]

# Combine w/ logic sum comp. star configs from peak and var. index filtering. 
new_config = ss.add(np.asarray([index_config,peak_config]))
print("Starting comp. star configuration:")
print(new_config)
print("\n")
used_info = fd.false_entries(total_filter)
info_update = 'Used information after filtering: %.2f / 100' % ((used_info*100.))
print(info_update)
print("\n")

# Computes all binned data in with different bin sizes, in the range
# bin_min - bin_max, with bin step of bin_interval. 
# Computes also optimal bin size for rms (RSD) minimization. 
if binning == True:
    print("Binning target rel. flux...")
    
    # Full binning configuration for generated .txt file. 
    bin_config = '%.2f_%.2f_%.2f' % ((bin_min,bin_max,bin_interval))
    # Base name:
    base_ = 'binsize_vs_rms_' + bin_config + '_' 
    # Complete file name:
    binrms_name = write.file_name(prefix_,base_,suffix_) + '.txt'
    binrms = open(binrms_name,"w+")
    
    # Compute rms for binned data for different bin sizes
    # from bin_min to bin_max with intervals of bin_interval.
    # Results are saved in file binrms_name.
    steps = int((bin_max - bin_min)/bin_interval)
    bin_rmss_list = []
    for i in range(0,steps + 1):
        # Set bin size
        bin_size = bin_min + bin_interval * float(i)
        
        # Obtain mean fluxes for each bin and get those
        # corresponding to the target. 
        #print('aqui no passa res')
        binned_fluxes = fd.bin_photometry(MDATA,bin_size,new_config)
        binned_target_flux = binned_fluxes[1][0]
        binned_target_flux_err = binned_fluxes[2][0] 
        # Mask for NaN values in binned_target_flux.
        # NaN values correspond to bins where there are no fluxes from 
        # original data. 
        binned_info        = np.isnan(binned_target_flux)
        # Number of not-NaN values in binned_target_flux
        nan_number_comp    = len(binned_info) - np.sum(binned_info)
        
        # Reduce binned_target_flux to np.array with no NaN values
        # and get corresponding rms. 
        mask_flux_err = np.isnan(binned_target_flux) + np.isnan(binned_target_flux_err)
        binned_target_flux     = binned_target_flux[~mask_flux_err]
        binned_target_flux_err = binned_target_flux_err[~mask_flux_err]
        binned_target_rms  = vi.rms(binned_target_flux,binned_target_flux_err,weighted)
        bin_rmss_list.append(binned_target_rms)
        binrms.write('%d \t %.6f\t %.10f\t %.2f \t %.2f\n' % (i,bin_size,binned_target_rms,float(nan_number_comp) / float(len(binned_info)) * 100.,nan_number_comp))
        bar.printProgressBar(i + 1, steps + 1, prefix_ = 'Progress:', suffix_ = 'Complete', length = 50)
    binrms.close()
    print("Done.")
    
    bin_rmss = np.asarray(bin_rmss_list)
    opt_bin_size = bin_min + bin_interval * float(np.argmin(bin_rmss))
    best_bin_rms = bin_rmss[np.argmin(bin_rmss)]
    
    bin_conclusion = 'Optimal bin size = %.2f (min. rms = %.6f)' % ((opt_bin_size,best_bin_rms)) 
    print(bin_conclusion)
    
    # Get bin fluxes from optimal bin size w.r.t. rms for both
    # target and comp. stars. 
    binned_fluxes = fd.bin_photometry(MDATA,opt_bin_size,new_config)
    binned_target_flux     = binned_fluxes[1][0]
    binned_target_flux_err = binned_fluxes[2][0]
    binned_comp_fluxes     = binned_fluxes[3]
    binned_comp_fluxes_err = binned_fluxes[4]
    
    # Get number of selected stars according to new_config
    c_new_config       = len(new_config) - np.sum(np.asarray(new_config))
    c_new_config       = int(c_new_config)
    
    # NaN values masks corresponding to both target and comp. stars
    # from binned_fluxes. 
    masks = binned_fluxes[5]
    t_total_nans = masks[0][0] + masks[1][0]
    c_total_nans = fd.mask_prod(masks[2]) + fd.mask_prod(masks[3])
    
    
    # Set total mask and obtain target and comp. star
    # rel. flux corresponding to optimal binning with no Nan values
    # for forecoming computations. 
    final_mask = t_total_nans + c_total_nans
    size = len(final_mask) - np.sum(final_mask)
    BDATA_T_ERR = np.zeros((1,size),dtype=np.float64)
    BDATA_T = np.zeros((1,size),dtype=np.float64)
    BDATA_T[0] = np.ma.masked_array(data=binned_target_flux,mask=final_mask).compressed()
    BDATA_T_ERR[0] = np.ma.masked_array(data=binned_target_flux_err,mask=final_mask).compressed()
    BDATA_C_ERR = np.zeros((c_new_config,size),dtype=np.float64)
    BDATA_C = np.zeros((c_new_config,size),dtype=np.float64)
    for i in range(0,c_new_config):
        BDATA_C[i]     = np.ma.masked_array(data=binned_comp_fluxes[i],mask=final_mask).compressed()
        BDATA_C_ERR[i] = np.ma.masked_array(data=binned_comp_fluxes_err[i],mask=final_mask).compressed()
        
TEST = open('test.txt',"w+")
negative = 0
for i in range(0,len(MDATA[6][0])):
    if MDATA[5][0][i] < 0: negative = 1
    else: negative = 0
    TEST.write('%.8f\t %d \n' % (MDATA[6][0][i],negative))
if c_selection == True:
    # Compute minimum rms and corresponding configurations, according to the method under consideration.         
    if binning == True: 
        print("Computing comp. star selection using binned + filtered data...")
        RESULTS_BINNED = ss.min_rms(BDATA_T,BDATA_T_ERR,BDATA_C,BDATA_C_ERR,method,new_config,k_max,N_min)
        # Setting filenames
        # Base names:
        f1_base = 'best_rmss_BINNED'
        f2_base = 'all_rmss_BINNED'
        g1_base = 'best_configs_BINNED'
        g2_base = 'all_configs_BINNED'

        # Complete file names:
        f1_name = write.file_name(prefix_,f1_base,suffix_) + '.txt'
        f2_name = write.file_name(prefix_,f2_base,suffix_) + '.txt'
        g1_name = write.file_name(prefix_,g1_base,suffix_) + '.txt'
        g2_name = write.file_name(prefix_,g2_base,suffix_) + '.txt'

        f1 = open(f1_name,"w+")
        f2 = open(f2_name,"w+")
        g1 = open(g1_name,"w+")
        g2 = open(g2_name,"w+")

        #size = len(RESULTS)
        m = int(np.sum(np.asarray(new_config)))

        for i in range(0,k_max - m):
            f1.write("%.2f\t %.8f\n" % (float(m + i + 1),RESULTS_BINNED[i][0][0][0]))
            g1.write("%.2f\t" % (float(m + i + 1)))
            for j in range(0,comp_stars):
                g1.write("%d" % RESULTS_BINNED[i][1][0][j])
                if j != comp_stars - 1: g1.write(",")
            g1.write("\n")
            
        f1.close()
        g1.close()

        for i in range(0,k_max - m):
            l = len(RESULTS_BINNED[i][0][0])
            for j in range(0,l):
                f2.write("%.2f\t %.8f\n" % (float(m + i + 1),RESULTS_BINNED[i][0][0][j]))
                g2.write("%.2f\t" % (float(m + i + 1)))
                for k in range(0,comp_stars):
                    g2.write("%d" % RESULTS_BINNED[i][1][j][k])
                    if k != comp_stars - 1: g2.write(",")
                g2.write("\n")

            f2.write("\n")
            g2.write("\n")

        f2.close()
        g2.close()
        print("\n")
        print("Computing comp. star selection using filtered...")
    else: 
        print("Computing comp. star selection using filtered...")
        RESULTS = ss.min_rms(MDATA[2],MDATA[3],MDATA[5],MDATA[6],method,new_config,k_max,N_min)

    print(RESULTS)
    # Setting filenames
    # Base names:
    f1_base = 'best_rmss'
    f2_base = 'all_rmss'
    g1_base = 'best_configs'
    g2_base = 'all_configs'

    # Complete file names:
    f1_name = write.file_name(prefix_,f1_base,suffix_) + '.txt'
    f2_name = write.file_name(prefix_,f2_base,suffix_) + '.txt'
    g1_name = write.file_name(prefix_,g1_base,suffix_) + '.txt'
    g2_name = write.file_name(prefix_,g2_base,suffix_) + '.txt'

    f1 = open(f1_name,"w+")
    f2 = open(f2_name,"w+")
    g1 = open(g1_name,"w+")
    g2 = open(g2_name,"w+")

    size = len(RESULTS)
    m = int(np.sum(np.asarray(new_config)))

    for i in range(0,k_max - m):
        f1.write("%.2f\t %.8f\n" % (float(m + i + 1),RESULTS[i][0][0][0]))
        g1.write("%.2f\t" % (float(m + i + 1)))
        for j in range(0,comp_stars):
            g1.write("%d" % RESULTS[i][1][0][j])
            if j != comp_stars - 1: g1.write(",")
        g1.write("\n")
        
    f1.close()
    g1.close()

    for i in range(0,k_max - m):
        l = len(RESULTS[i][0][0])
        for j in range(0,l):
            f2.write("%.2f\t %.8f\n" % (float(m + i + 1),RESULTS[i][0][0][j]))
            g2.write("%.2f\t" % (float(m + i + 1)))
            for k in range(0,comp_stars):
                g2.write("%d" % RESULTS[i][1][j][k])
                if k != comp_stars - 1: g2.write(",")
            g2.write("\n")

        f2.write("\n")
        g2.write("\n")

    f2.close()
    g2.close()

# Star selection with minimization of the snr index in varindex.py
if c_selection_snr == True:       
    if binning == True: 
        print("Computing comp. star selection using binned + filtered data...")
        RESULTS_BINNED = ss.min_snr(BDATA_T,BDATA_T_ERR,BDATA_C,BDATA_C_ERR,method,new_config,k_max,N_min)
        # Setting filenames
        # Base names:
        f1_base = 'best_l1_BINNED'
        f2_base = 'all_l1_BINNED'
        g1_base = 'best_configs_l1_BINNED'
        g2_base = 'all_configs_l1_BINNED'

        # Complete file names:
        f1_name = write.file_name(prefix_,f1_base,suffix_) + '.txt'
        f2_name = write.file_name(prefix_,f2_base,suffix_) + '.txt'
        g1_name = write.file_name(prefix_,g1_base,suffix_) + '.txt'
        g2_name = write.file_name(prefix_,g2_base,suffix_) + '.txt'

        f1 = open(f1_name,"w+")
        f2 = open(f2_name,"w+")
        g1 = open(g1_name,"w+")
        g2 = open(g2_name,"w+")

        m = int(np.sum(np.asarray(new_config)))

        for i in range(0,k_max - m):
            f1.write("%.2f\t %.8f\n" % (float(m + i + 1),RESULTS_BINNED[i][0][0][0]))
            g1.write("%.2f\t" % (float(m + i + 1)))
            for j in range(0,comp_stars):
                g1.write("%d" % RESULTS_BINNED[i][1][0][j])
                if j != comp_stars - 1: g1.write(",")
            g1.write("\n")
            
        f1.close()
        g1.close()

        for i in range(0,k_max - m):
            l = len(RESULTS_BINNED[i][0][0])
            for j in range(0,l):
                f2.write("%.2f\t %.8f\n" % (float(m + i + 1),RESULTS_BINNED[i][0][0][j]))
                g2.write("%.2f\t" % (float(m + i + 1)))
                for k in range(0,comp_stars):
                    g2.write("%d" % RESULTS_BINNED[i][1][j][k])
                    if k != comp_stars - 1: g2.write(",")
                g2.write("\n")

            f2.write("\n")
            g2.write("\n")

        f2.close()
        g2.close()
        print('\n')
        print("Computing comp. star selection using filtered data...")
        RESULTS = ss.min_rms(MDATA[2],MDATA[3],MDATA[5],MDATA[6],method,new_config,k_max,N_min)
    else: RESULTS = ss.min_snr(MDATA[2],MDATA[3],MDATA[5],MDATA[6],method,new_config,k_max,N_min)
    
    # Setting filenames
    # Base names:
    f1_base = 'best_l1'
    f2_base = 'all_l1'
    g1_base = 'best_configs_l1'
    g2_base = 'all_configs_l1'

    # Complete file names:
    f1_name = write.file_name(prefix_,f1_base,suffix_) + '.txt'
    f2_name = write.file_name(prefix_,f2_base,suffix_) + '.txt'
    g1_name = write.file_name(prefix_,g1_base,suffix_) + '.txt'
    g2_name = write.file_name(prefix_,g2_base,suffix_) + '.txt'

    f1 = open(f1_name,"w+")
    f2 = open(f2_name,"w+")
    g1 = open(g1_name,"w+")
    g2 = open(g2_name,"w+")

    m = int(np.sum(np.asarray(new_config)))

    for i in range(0,k_max - m):
        f1.write("%.2f\t %.8f\n" % (float(m + i + 1),RESULTS[i][0][0][0]))
        g1.write("%.2f\t" % (float(m + i + 1)))
        for j in range(0,comp_stars):
            g1.write("%d" % RESULTS[i][1][0][j])
            if j != comp_stars - 1: g1.write(",")
        g1.write("\n")
        
    f1.close()
    g1.close()

    for i in range(0,k_max - m):
        l = len(RESULTS[i][0][0])
        for j in range(0,l):
            f2.write("%.2f\t %.8f\n" % (float(m + i + 1),RESULTS[i][0][0][j]))
            g2.write("%.2f\t" % (float(m + i + 1)))
            for k in range(0,comp_stars):
                g2.write("%d" % RESULTS[i][1][j][k])
                if k != comp_stars - 1: g2.write(",")
            g2.write("\n")

        f2.write("\n")
        g2.write("\n")

    f2.close()
    g2.close()




    
    
