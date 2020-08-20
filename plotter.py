# DIFFERENT PLOT IMPLEMENTATIONS FOR REPRESENTATION
# OF RESULTS USING FILTERING AND STAR SELECTION OPTIMIZATION 
# PROCEDURES. 

# The following type of plots can be generated with this
# program:
#   * Binned fluxes 
#   * Binned phase-fold diagrams
#   * RMS (RSD) vs # of selected stars (target RSD minimization)
#   * RMS (RSD) vs. bin size
#   * Raw vs. filtered comparison plots (with flux, SNR and periodogram graphics)
#   * Phase-fold of filtered/raw fluxes
#   * Discarded comp. star fluxes with normalized index bar charts
#   * Multiple LS periodograms from iterative prewhitening

import math
import numpy as np
from astropy.stats import sigma_clip
from astropy.timeseries import LombScargle
from astropy.timeseries import TimeSeries
import sys
sys.path.append('/home/guifress/pdo/modules')
import read
import write
import filter_data as fd
import starselection as ss
import more_stats as st
import varindex as vi

# matplotlib config. 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.offsetbox import AnchoredText
'''
params = {'legend.fontsize': 11,
          'figure.figsize': (15, 5),
         'axes.labelsize': 14,
         'axes.titlesize': 11,
         'xtick.labelsize': 11,
         'ytick.labelsize': 11}
pylab.rcParams.update(params)
'''
dot_size = 2.

extension = '.pdf'

from matplotlib import rcParams
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "cm"
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
org    = [2739,1,15,17]
target = [22,33]
comp   = [23,24,25,26,27,28,29,30,31,32,34,35,36,37,38]
ind    = [target,comp]
'''


# Data organization in TZAri file:
# org[0] = 3455 (cropped file), org[0] = 3646 (full file)
org    = [3646,1,15,17]
target = [20,33]
comp   = [21,22,23,24,25,26,27,28,29,30,31,32,34,35,36]
ind    = [target,comp]


'''
# Data organization in Wolf1069 file:
org    = [653,1,15,22]
target = [20,24,25,27,31,32,33]
comp   = [21,22,23,26,28,29,30,34,35,36,37,38,39,40,41]
ind    = [target,comp]
'''

comp_stars1 = 15.
comp_stars2 = 15.
comp_stars3 = 13.
comp_stars4 = 15.

# Specify filters that are to be used.
threshold_filter = True
c_filter         = True
t_filter         = True
peak_filter      = True
var_index        = True
binning          = False
c_selection      = False
c_selection_snr  = False

# Sigma clip and snr numerical clip parameters for comp. star filtering:
c_su = 4.
c_sl = 4.
c_p  = 0.
c_m  = True

# Sigma clip and snr numerical clip parameters for target filtering: 
t_su = 3.
t_sl = 3.
t_p  = 0.35
t_m  = True

peak_limit = 45000.

# Options, parameters and general config. for computation of variability indices:
opts       = [True,True,True,True]
arr        = [True,False,False,True]
ks         = [10,1,1,10]
thresholds = [1.,1.,1.,0.1]
for i in range(0,4): 
    if arr[i] == True: var_index = True

# Binning parameters
comp_stars = len(comp)
bin_min  = .25 * 1. / 24.
bin_max  =  2.
bin_interval = .5 * .25 * 1. / 24.

# Parameters and method for comp. star selection:
comp_stars = len(comp)
k_max = comp_stars - 1
N_min = 5
method = False

# Filename suffix and prefix:
# Suffix: 
suffix_ = write.suffix(c_filter,c_m,c_su,c_sl,c_p,t_filter,t_m,t_su,t_sl,t_p,var_index,arr,opts,ks,thresholds,method,k_max,N_min,peak_limit,binning,bin_interval)
# Prefix:
prefix_ = 'TZAri_EXAMPLE_'

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
    print("\t",index_config)
    print("\t Done.")
    print("\n") 
else:
    index_config = [0 for _ in range(0,comp_stars)]

# Combine w/ logic sum comp. star configs from peak and var. index filtering. 
new_config = ss.add(np.asarray([index_config,peak_config]))
print("\t",new_config)
used_info = fd.false_entries(total_filter)
info_update = 'Used information after filtering: %.2f / 100' % ((used_info*100.))
print(info_update)
print("\n")

# Comp. star configuration that will be used throughout the plot 
opt_config = new_config
# Example opt_configs
#opt_config = [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
#opt_config = [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1]
common_width = 12.


# Specify which plots are to be generated. 
BINNED_FLUX             = False
BINNED_PERIODOGRAM      = False
BINNED_PHASE_FOLDING    = False
RMS_VS_SELECTED         = False
RMS_VS_BINSIZE          = False
RAW_FILTERED_COMPARISON = False
PHASE_FOLDING           = False 
COMP_STAR_VAR_INDICES   = False 
PREWHITENING            = True
print("Generating plot(s)...")

# General plotting parameters
max_freq = 0.6
maxBJD = CDATA[0][0].max()
minBJD = CDATA[0][0].min()
min_freq_factor = 1e-5 * (maxBJD - minBJD)
bin_size = 1. / 24.
n0 = 100.
period_terms = 1
amp = 3.5
prefix_ += '%.2f_' % (bin_size*24.)

# Summary of selected parameters.
print("Filtering parameters (alpha_t,p_t,alpha_c,p_c): %.6f,\t %.6f,\t %.6f,\t %.6f" % (t_su,t_p,c_su,c_p))
print("Comp. star configuration:",opt_config)
print("Max. freq. in LS periodogram: %.6f" % (max_freq))
print("Min. freq. in LS periodogram (and min_freq_factor): %.6f (%.6f)" % ((min_freq_factor * 1. / (maxBJD-minBJD),min_freq_factor)))
print("freq. grid factor: %.2f" % (n0))

if RMS_VS_SELECTED == True:
    f1_base = 'best_rmss_BINNED'
    f1_name = write.file_name(prefix_,f1_base,suffix_) + '.txt'

    f1_base_filtered = 'best_rmss'
    f1_name_filtered = write.file_name(prefix_,f1_base_filtered,suffix_) + '.txt'

    #filtered_data_filename = 'GJ555_best_l1_filtcomp_mean_4.0_4.0_0.0_filttarget_mean_3.0_3.0_0.2_novarindex_dynamic_kmax_14_pl_45000.0_binned_0.01.txt'
    #binned_data_filename   = 'GJ555_best_l1_BINNED_filtcomp_mean_4.0_4.0_0.0_filttarget_mean_3.0_3.0_0.2_novarindex_dynamic_kmax_14_pl_45000.0_binned_0.01.txt'

    raw_data_filename1      = 'TZAri_cropped_best_rmss_cropped.txt'
    raw_data_filename2      = 'GJ555_cropped_3_best_rmss.txt'
    raw_data_filename3      = 'TOI-1266_best_rmss.txt'
    raw_data_filename4      = 'Wolf1069_best_rmss.txt'

if PREWHITENING == True:
    faps_probs = [10./100.,1./100.,0.1/100.]
    s = 1.25
    params = {'legend.fontsize': 9.*s,
            'axes.labelsize': 12*s,
            'axes.titlesize': 12*s,
            'xtick.labelsize': 10*s,
            'ytick.labelsize': 10*s}
    pylab.rcParams.update(params)
    dot_size = 2.15*s
    # Raw data vs. filtered data comparison
    # Get non-discarded data from previous filtering.
    T1  = CDATA[2][0]
    DT1 = CDATA[3][0]
    original_config = [0 for _ in range(0,comp_stars)]
    MT1 = ss.new_flux(MDATA[2],MDATA[5],opt_config)[0][0]
    DMT1 = ss.new_flux_err(MDATA[2],MDATA[3],MDATA[5],MDATA[6],opt_config)[0][0]
    BJD = CDATA[0][0]
    MBJD = MDATA[0][0]
    OMT1 = ss.new_flux(MDATA[2],MDATA[5],original_config)[0][0]
    ODMT1 = ss.new_flux_err(MDATA[2],MDATA[3],MDATA[5],MDATA[6],original_config)[0][0]
    OMBJD = MDATA[0][0]
    
    
    ax1 = plt.subplot(421)
    DT1 = CDATA[3][0]
    DMDATA = ss.new_flux_err(MDATA[2],MDATA[3],MDATA[5],MDATA[6],opt_config)
    DMT1 = DMDATA[0][0]
    frequency  = np.arange(min_freq_factor * 1. / ( BJD.max() -  BJD.min()), max_freq ,1. / (n0 * ( BJD.max() -  BJD.min())))
    Mfrequency = np.arange(min_freq_factor * 1. / ( BJD.max() -  BJD.min()), max_freq ,1. / (n0 * ( BJD.max() -  BJD.min())))
    power  = LombScargle( BJD, T1, DT1,normalization='standard',nterms=period_terms).power(frequency)
    Mpower = LombScargle(MBJD,MT1,DMT1,normalization='standard',nterms=period_terms).power(Mfrequency)
    OMpower = LombScargle(OMBJD,OMT1,ODMT1,normalization='standard',nterms=period_terms).power(Mfrequency)
    MLS    = LombScargle(MBJD,MT1,DMT1,normalization='standard',nterms=period_terms)
    OMLS   = LombScargle(OMBJD,OMT1,ODMT1,normalization='standard',nterms=period_terms)
    RLS    = LombScargle( BJD, T1, DT1,normalization='standard',nterms=period_terms)
    nu_1      =  frequency[np.argmax(power )]
    Mnu_1     = Mfrequency[np.argmax(Mpower)]
    OMnu_1    = Mfrequency[np.argmax(OMpower)]
    Period_1  = 1. /  frequency[np.argmax(power )]
    MPeriod_1 = 1. / Mfrequency[np.argmax(Mpower)]
    OMPeriod_1 = 1. / OMnu_1
    label1 = r'$\mathrm{raw \ data \ (main \ period = %.2f \ d, \ freq. = %.4f \ d}^{-1})$' % ((Period_1,nu_1))
    label2 = r'$\mathrm{filtered \ data} \   \ \mathrm{(main \ period = %.2f \ d, \ freq. = %.4f \ d}^{-1})$' % ((MPeriod_1,Mnu_1))
    plt.axvline(x=nu_1,color='blue',linestyle='--',linewidth=1.)
    plt.axvline(x=Mnu_1,color='gray',linestyle='--',linewidth=1.)
    Rlevels = RLS.false_alarm_level(faps_probs)  
    Mlevels = MLS.false_alarm_level(faps_probs)
    #label=r'$\mathrm{FAP \ %.3f}$'% (faps_probs[0]*100.) + r'$\%$'
    plt.axhline(y=Mlevels[0],color='black',linestyle='dotted',linewidth=1.4,)
    plt.axhline(y=Mlevels[1],color='black',linestyle='dashdot',linewidth=1.4)
    plt.axhline(y=Mlevels[2],color='black',linestyle='solid',linewidth=1.4)
    #for i in range(2,4):
    #    plt.axvline(x=Mnu_1 / float(i),color='black',linestyle='--',linewidth=1.)
    ax1.set_xlim([0.,max_freq])
    ax1.plot(frequency,power,linestyle='-',lw=1,marker='o',color='blue',markersize=0.0,label=label1)
    ax1.plot(Mfrequency,Mpower,linestyle='-',lw=1,marker='o',color='gray',markersize=0.0,label=label2)
    plt.xlabel(r'$\nu \ (\mathrm{d}^{-1})$')
    plt.ylabel(r'$\mathrm{Power}$')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    ax2 = plt.subplot(422)
    label1 = r'$\mathrm{LS \ model \ best-fit \ at \ }T_{\mathrm{peak}} = %.2f \ \mathrm{days}$' % (MPeriod_1)
    label2 = r'$\mathrm{filtered \ data} \  $'
    Mt_fit = np.linspace(MBJD.min(),MBJD.max(),5000) 
    My_fit = MLS.model(Mt_fit,Mnu_1)
    theta = MLS.model_parameters(Mfrequency[np.argmax(Mpower)])
    offset = MLS.offset()
    design_matrix = MLS.design_matrix(Mnu_1,Mt_fit)
    modelys = offset + design_matrix.dot(theta)
    ax2.plot(MBJD,MT1,marker='o',markersize=dot_size,label=label2,color='gray',linestyle='None')
    ax2.plot(Mt_fit,modelys,color='black',linestyle='-',lw=1.*s,label = label1)
    plt.xlabel(r'$\mathrm{BJD }$')
    plt.ylabel(r'$\mathrm{target \ rel. \ flux}$')
    plt.grid(True)
    plt.legend(loc='upper right')
    # PREWHITENING RESULTS
    print("PW 0")
    print("Max. freq.:")
    print("Raw \t %.8f; Filtered \t %.8f" % (nu_1,Mnu_1))
    print("Max. powers:")
    print("Raw \t %.8f; Filtered \t %.8f" % (np.amax(power),np.amax(Mpower)))
    print("False Alarm Probabilities at peak powers:")
    print("Raw data:\t",RLS.false_alarm_probability(power.max()))
    print("Filtered data:\t",MLS.false_alarm_probability(Mpower.max()))
    print("\n")
    print("FAP levels:")
    print("Raw data:\t",Rlevels)
    print("Filtered data:\t",Mlevels)
    print("Lomb-Scargle filtered data best-fit parameters:")
    print(theta)
    print("\n")
    
    ax3 = plt.subplot(423)
    DT1 = CDATA[3][0]
    DMDATA = ss.new_flux_err(MDATA[2],MDATA[3],MDATA[5],MDATA[6],opt_config)
    DMT1 = DMDATA[0][0]
    
    theta = MLS.model_parameters(Mfrequency[np.argmax(Mpower)])
    offset = MLS.offset()
    design_matrix = MLS.design_matrix(Mnu_1,MBJD)
    modelys = offset + design_matrix.dot(theta)
    MT1 = MT1 - np.asarray(modelys)
    
    Otheta = OMLS.model_parameters(Mfrequency[np.argmax(OMpower)])
    Ooffset = OMLS.offset()
    Odesign_matrix = OMLS.design_matrix(OMnu_1,OMBJD)
    Omodelys = Ooffset + Odesign_matrix.dot(Otheta)
    OMT1 = OMT1 - np.asarray(Omodelys)
    
    frequency  = np.arange(min_freq_factor * 1. / ( BJD.max() -  BJD.min()), max_freq ,1. / (n0 * ( BJD.max() -  BJD.min())))
    Mfrequency = np.arange(min_freq_factor * 1. / ( BJD.max() -  BJD.min()), max_freq ,1. / (n0 * ( BJD.max() -  BJD.min())))
    power  = LombScargle( BJD, T1, DT1,normalization='standard',nterms=period_terms).power(frequency)
    Mpower = LombScargle(MBJD,MT1,normalization='standard',nterms=period_terms).power(Mfrequency)
    MLS    = LombScargle(MBJD,MT1,normalization='standard',nterms=period_terms)
    OMLS   = LombScargle(OMBJD,OMT1,normalization='standard',nterms=period_terms)
    RLS    = LombScargle( BJD, T1, DT1,normalization='standard',nterms=period_terms)
    nu_1      =  frequency[np.argmax(power )]
    Mnu_1     = Mfrequency[np.argmax(Mpower)]
    Period_1  = 1. /  frequency[np.argmax(power )]
    MPeriod_1 = 1. / Mfrequency[np.argmax(Mpower)]
    label1 = r'$\mathrm{raw \ data \ (main \ period = %.2f \ d, \ freq. = %.4f \ d}^{-1})$' % ((Period_1,nu_1))
    label2 = r'$\mathrm{filtered \ data} \   \ \mathrm{(main \ period = %.2f \ d, \ freq. = %.4f \ d}^{-1})$' % ((MPeriod_1,Mnu_1))
    #plt.axvline(x=nu_1,color='blue',linestyle='--',linewidth=1.)
    plt.axvline(x=Mnu_1,color='gray',linestyle='--',linewidth=1.)
    Rlevels = RLS.false_alarm_level(faps_probs)  
    Mlevels = MLS.false_alarm_level(faps_probs)
    #label=r'$\mathrm{FAP \ %.3f}$'% (faps_probs[0]*100.) + r'$\%$'
    plt.axhline(y=Mlevels[0],color='black',linestyle='dotted',linewidth=1.4,)
    plt.axhline(y=Mlevels[1],color='black',linestyle='dashdot',linewidth=1.4)
    plt.axhline(y=Mlevels[2],color='black',linestyle='solid',linewidth=1.4)
    #for i in range(2,4):
    #    plt.axvline(x=Mnu_1 / float(i),color='black',linestyle='--',linewidth=1.)
    ax3.set_xlim([0.,max_freq])
    ax3.plot(Mfrequency,Mpower,linestyle='-',lw=1,marker='o',color='gray',markersize=0.0,label=label2)
    plt.xlabel(r'$\nu \ (\mathrm{d}^{-1})$')
    plt.ylabel(r'$\mathrm{Power}$')
    plt.grid(True)
    plt.legend(loc='upper left')
    F2T1 = MT1 
    F2nu_1 = Mnu_1
    F2LS = MLS
    OF2LS = OMLS

    ax4 = plt.subplot(424)
    label1 = r'$\mathrm{LS \ model \ best-fit \ at \ }T_{\mathrm{peak}} = %.2f \ \mathrm{days}$' % (MPeriod_1)
    label2 = r'$\mathrm{filtered \ data} \  $'
    Mt_fit = np.linspace(MBJD.min(),MBJD.max(),5000) 
    My_fit = MLS.model(Mt_fit,Mnu_1)
    theta = MLS.model_parameters(Mfrequency[np.argmax(Mpower)])
    offset = MLS.offset()
    design_matrix = MLS.design_matrix(Mnu_1,Mt_fit)
    modelys = offset + design_matrix.dot(theta)
    ax4.plot(MBJD,MT1,marker='o',markersize=dot_size,label=label2,color='gray',linestyle='None')
    ax4.plot(Mt_fit,modelys,color='black',linestyle='-',lw=1.*s,label = label1)
    plt.xlabel(r'$\mathrm{BJD }$')
    plt.ylabel(r'$\mathrm{target \ (shifted) \ rel. \ flux}$')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    print("PW 1")
    print("Max. freq.:")
    print("Raw \t %.8f; Filtered \t %.8f" % (nu_1,Mnu_1))
    print("Max. powers:")
    print("Raw \t %.8f; Filtered \t %.8f" % (np.amax(power),np.amax(Mpower)))
    print("False Alarm Probabilities at peak powers:")
    print("Raw data:\t",RLS.false_alarm_probability(power.max()))
    print("Filtered data:\t",MLS.false_alarm_probability(Mpower.max()))
    print("\n")
    print("FAP levels:")
    print("Raw data:\t",Rlevels)
    print("Filtered data:\t",Mlevels)
    print("Lomb-Scargle filtered data best-fit parameters:")
    print(theta)
    print("\n")
    
    ax5 = plt.subplot(425)
    DT1 = CDATA[3][0]
    DMDATA = ss.new_flux_err(MDATA[2],MDATA[3],MDATA[5],MDATA[6],opt_config)
    DMT1 = DMDATA[0][0]

    theta = MLS.model_parameters(Mfrequency[np.argmax(Mpower)])
    offset = MLS.offset()
    design_matrix = MLS.design_matrix(Mnu_1,MBJD)
    modelys = offset + design_matrix.dot(theta)
    MT1 = MT1 - np.asarray(modelys)
    
    frequency  = np.arange(min_freq_factor * 1. / ( BJD.max() -  BJD.min()), max_freq ,1. / (n0 * ( BJD.max() -  BJD.min())))
    Mfrequency = np.arange(min_freq_factor * 1. / ( BJD.max() -  BJD.min()), max_freq ,1. / (n0 * ( BJD.max() -  BJD.min())))
    power  = LombScargle( BJD, T1, DT1,normalization='standard',nterms=period_terms).power(frequency)
    Mpower = LombScargle(MBJD,MT1,normalization='standard',nterms=period_terms).power(Mfrequency)
    MLS    = LombScargle(MBJD,MT1,normalization='standard',nterms=period_terms)
    RLS    = LombScargle( BJD, T1, DT1,normalization='standard',nterms=period_terms)
    nu_1      =  frequency[np.argmax(power )]
    Mnu_1     = Mfrequency[np.argmax(Mpower)]
    Period_1  = 1. /  frequency[np.argmax(power )]
    MPeriod_1 = 1. / Mfrequency[np.argmax(Mpower)]
    label1 = r'$\mathrm{raw \ data \ (main \ period = %.2f \ d, \ freq. = %.4f \ d}^{-1})$' % ((Period_1,nu_1))
    label2 = r'$\mathrm{filtered \ data} \   \ \mathrm{(main \ period = %.2f \ d, \ freq. = %.4f \ d}^{-1})$' % ((MPeriod_1,Mnu_1))
    #plt.axvline(x=nu_1,color='blue',linestyle='--',linewidth=1.)
    plt.axvline(x=Mnu_1,color='gray',linestyle='--',linewidth=1.)
    Rlevels = RLS.false_alarm_level(faps_probs)  
    Mlevels = MLS.false_alarm_level(faps_probs)
    #label=r'$\mathrm{FAP \ %.3f}$'% (faps_probs[0]*100.) + r'$\%$'
    plt.axhline(y=Mlevels[0],color='black',linestyle='dotted',linewidth=1.4,)
    plt.axhline(y=Mlevels[1],color='black',linestyle='dashdot',linewidth=1.4)
    plt.axhline(y=Mlevels[2],color='black',linestyle='solid',linewidth=1.4)
    #for i in range(2,4):
    #    plt.axvline(x=Mnu_1 / float(i),color='black',linestyle='--',linewidth=1.)
    ax5.set_xlim([0.,max_freq])
    ax5.plot(Mfrequency,Mpower,linestyle='-',lw=1,marker='o',color='gray',markersize=0.0,label=label2)
    plt.xlabel(r'$\nu \ (\mathrm{d}^{-1})$')
    plt.ylabel(r'$\mathrm{Power}$')
    plt.grid(True)
    plt.legend(loc='upper right')
    F3T1 = MT1 
    F3nu_1 = Mnu_1
    F3LS = MLS
    ax6 = plt.subplot(426)
    label1 = r'$\mathrm{LS \ model \ best-fit \ at \ }T_{\mathrm{peak}} = %.2f \ \mathrm{days}$' % (MPeriod_1)
    label2 = r'$\mathrm{filtered \ data} \  $'
    Mt_fit = np.linspace(MBJD.min(),MBJD.max(),5000) 
    My_fit = MLS.model(Mt_fit,Mnu_1)
    theta = MLS.model_parameters(Mfrequency[np.argmax(Mpower)])
    offset = MLS.offset()
    design_matrix = MLS.design_matrix(Mnu_1,Mt_fit)
    modelys = offset + design_matrix.dot(theta)
    ax6.plot(MBJD,MT1,marker='o',markersize=dot_size,label=label2,color='gray',linestyle='None')
    ax6.plot(Mt_fit,modelys,color='black',linestyle='-',lw=1.*s,label = label1)
    plt.xlabel(r'$\mathrm{BJD }$')
    plt.ylabel(r'$\mathrm{target \ (shifted) \ rel. \ flux}$')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    print("PW 2")
    print("Max. freq.:")
    print("Raw \t %.8f; Filtered \t %.8f" % (nu_1,Mnu_1))
    print("Max. powers:")
    print("Raw \t %.8f; Filtered \t %.8f" % (np.amax(power),np.amax(Mpower)))
    print("False Alarm Probabilities at peak powers:")
    print("Raw data:\t",RLS.false_alarm_probability(power.max()))
    print("Filtered data:\t",MLS.false_alarm_probability(Mpower.max()))
    print("\n")
    print("FAP levels:")
    print("Raw data:\t",Rlevels)
    print("Filtered data:\t",Mlevels)
    print("Lomb-Scargle filtered data best-fit parameters:")
    print(theta)
    print("\n")
    
    ax7 = plt.subplot(427)
    DT1 = CDATA[3][0]
    DMDATA = ss.new_flux_err(MDATA[2],MDATA[3],MDATA[5],MDATA[6],opt_config)
    DMT1 = DMDATA[0][0]
    
    theta = MLS.model_parameters(Mfrequency[np.argmax(Mpower)])
    offset = MLS.offset()
    design_matrix = MLS.design_matrix(Mnu_1,MBJD)
    modelys = offset + design_matrix.dot(theta)
    MT1 = MT1 - np.asarray(modelys)
    
    frequency  = np.arange(min_freq_factor * 1. / ( BJD.max() -  BJD.min()), max_freq ,1. / (n0 * ( BJD.max() -  BJD.min())))
    Mfrequency = np.arange(min_freq_factor * 1. / ( BJD.max() -  BJD.min()), max_freq ,1. / (n0 * ( BJD.max() -  BJD.min())))
    power  = LombScargle( BJD, T1, DT1,normalization='standard',nterms=period_terms).power(frequency)
    Mpower = LombScargle(MBJD,MT1,normalization='standard',nterms=period_terms).power(Mfrequency)
    MLS    = LombScargle(MBJD,MT1,normalization='standard',nterms=period_terms)
    RLS    = LombScargle( BJD, T1, DT1,normalization='standard',nterms=period_terms)
    nu_1      =  frequency[np.argmax(power )]
    Mnu_1     = Mfrequency[np.argmax(Mpower)]
    Period_1  = 1. /  frequency[np.argmax(power )]
    MPeriod_1 = 1. / Mfrequency[np.argmax(Mpower)]
    label1 = r'$\mathrm{raw \ data \ (main \ period = %.2f \ d, \ freq. = %.4f \ d}^{-1})$' % ((Period_1,nu_1))
    label2 = r'$\mathrm{filtered \ data} \   \ \mathrm{(main \ period = %.2f \ d, \ freq. = %.4f \ d}^{-1})$' % ((MPeriod_1,Mnu_1))
    #plt.axvline(x=nu_1,color='blue',linestyle='--',linewidth=1.)
    plt.axvline(x=Mnu_1,color='gray',linestyle='--',linewidth=1.)
    Rlevels = RLS.false_alarm_level(faps_probs)  
    Mlevels = MLS.false_alarm_level(faps_probs)
    #label=r'$\mathrm{FAP \ %.3f}$'% (faps_probs[0]*100.) + r'$\%$'
    plt.axhline(y=Mlevels[0],color='black',linestyle='dotted',linewidth=1.4,)
    plt.axhline(y=Mlevels[1],color='black',linestyle='dashdot',linewidth=1.4)
    plt.axhline(y=Mlevels[2],color='black',linestyle='solid',linewidth=1.4)
    #for i in range(2,4):
    #    plt.axvline(x=Mnu_1 / float(i),color='black',linestyle='--',linewidth=1.)
    ax7.set_xlim([0.,max_freq])
    #ax3.plot(frequency,power,linestyle='-',lw=1,marker='o',color='blue',markersize=0.0,label=label1)
    ax7.plot(Mfrequency,Mpower,linestyle='-',lw=1,marker='o',color='gray',markersize=0.0,label=label2)
    plt.xlabel(r'$\nu \ (\mathrm{d}^{-1})$')
    plt.ylabel(r'$\mathrm{Power}$')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    F4T1 = MT1 
    F4nu_1 = Mnu_1
    F4LS = MLS
    
    ax8 = plt.subplot(428)
    label1 = r'$\mathrm{LS \ model \ best-fit \ at \ }T_{\mathrm{peak}} = %.2f \ \mathrm{days}$' % (MPeriod_1)
    label2 = r'$\mathrm{filtered \ data} \  $'
    Mt_fit = np.linspace(MBJD.min(),MBJD.max(),5000) 
    My_fit = MLS.model(Mt_fit,Mnu_1)
    
    theta = MLS.model_parameters(Mfrequency[np.argmax(Mpower)])
    offset = MLS.offset()
    design_matrix = MLS.design_matrix(Mnu_1,Mt_fit)
    modelys = offset + design_matrix.dot(theta)
    
    ax8.plot(MBJD,MT1,marker='o',markersize=dot_size,label=label2,color='gray',linestyle='None')
    ax8.plot(Mt_fit,modelys,color='black',linestyle='-',lw=1.*s,label = label1)
    plt.xlabel(r'$\mathrm{BJD }$')
    plt.ylabel(r'$\mathrm{target \ (shifted) \ rel. \ flux}$')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    print("PW 3")
    print("Max. freq.:")
    print("Raw \t %.8f; Filtered \t %.8f" % (nu_1,Mnu_1))
    print("Max. powers:")
    print("Raw \t %.8f; Filtered \t %.8f" % (np.amax(power),np.amax(Mpower)))
    print("False Alarm Probabilities at peak powers:")
    print("Raw data:\t",RLS.false_alarm_probability(power.max()))
    print("Filtered data:\t",MLS.false_alarm_probability(Mpower.max()))
    print("\n")
    print("FAP levels:")
    print("Raw data:\t",Rlevels)
    print("Filtered data:\t",Mlevels)
    print("Lomb-Scargle filtered data best-fit parameters:")
    print(theta)
    print("\n")
    
    
    fig = plt.gcf() # get current figure
    fig.set_size_inches(s*common_width*1.25,s*2.15*4)
    plt.tight_layout()
    # Base name:
    base_ = 'multiple_LS'
    # Complete file names:
    plot_name = write.file_name(prefix_,base_,suffix_) + extension
    fig.savefig(plot_name, dpi=400, bbox_inches='tight')
    end = 'Plot (multiple LS periodograms) saved as:\n' + plot_name + '\n'
    print(end)  
    
if BINNED_FLUX == True:
    # BINNED DATA VS. RAW / FILTERED DATA
    T1  = CDATA[2][0]
    MT1 = ss.new_flux(MDATA[2],MDATA[5],opt_config)[0][0]
    DMT1 = ss.new_flux_err(MDATA[2],MDATA[3],MDATA[5],MDATA[6],opt_config)[0][0]
    BJD = CDATA[0][0]
    MBJD = MDATA[0][0]

    binned = fd.to_bin(MBJD,MT1,DMT1,bin_size)
    binned_T1  = binned[0]
    binned_T1_err = binned[1]
    mask_flux_err = np.isnan(binned[0]) + np.isnan(binned[1])
    binned_BJD = binned[2]

    dev_T1   = np.std(T1,dtype=np.float64)
    mean_T1  = np.mean(T1,dtype=np.float64)
    dev_T1  /= 1.
    mean_T1 /= 1.

    dev_MT1   = np.std(MT1,dtype=np.float64)
    mean_MT1  = np.mean(MT1,dtype=np.float64)
    dev_MT1  /= 1.
    mean_MT1 /= 1.

    dev_binned_T1   = np.std(binned_T1[~np.isnan(binned_T1)],dtype=np.float64)
    mean_binned_T1  = np.mean(binned_T1[~np.isnan(binned_T1)],dtype=np.float64)
    dev_binned_T1  /= 1.
    mean_binned_T1 /= 1.

    ax1 = plt.subplot(111)
    ax1.set_ylim([MT1.min() - .1 * (MT1.max() - MT1.min()),MT1.max() + .1 * (MT1.max() - MT1.min())])
    plt.axhline(y=mean_MT1,color='gray',linestyle='-',linewidth=1)
    plt.axhline(y=mean_MT1-t_sl*dev_MT1,color='gray',linestyle='--',linewidth=1)
    plt.axhline(y=mean_MT1+t_su*dev_MT1,color='gray',linestyle='--',linewidth=1)

    plt.axhline(y=mean_binned_T1,color='black',linestyle='-',linewidth=1)
    plt.axhline(y=mean_binned_T1-dev_binned_T1,color='black',linestyle='--',linewidth=1)
    plt.axhline(y=mean_binned_T1+dev_binned_T1,color='black',linestyle='--',linewidth=1)

    label1 = r'$\mathrm{filtered \ data} \ (\pm %.2f\sigma, \ p = %.2f): \ \mathrm{rms = %.6f}$' % ((t_su,t_p,vi.rms(MT1,DMT1,weighted)))
    label2 = r'$\mathrm{filtered} + \mathrm{binned \ data} \ (\mathrm{bin \ size}  = %.2f \ \mathrm{h}): \ \mathrm{rms = %.6f}$' % ((bin_size * 24.,vi.rms(binned_T1[~mask_flux_err],binned_T1_err[~mask_flux_err],weighted)))

    ax1.plot(MBJD,MT1,marker='o',color='gray',markersize=dot_size,label=label1,linestyle='None')
    #ax1.errorbar(BJD,T1,yerr=DATA[3][0,:]*1.,marker='o',color='red',markersize=1,label=label1,linestyle='None',capsize=5)
    #ax1.plot(binned_BJD,binned_T1,marker='o',markersize=dot_size*2.,label=label2,color='black',linestyle='None')
    ax1.errorbar(binned_BJD,binned_T1,yerr=binned[1],marker='o',color='black',markersize=dot_size*2,label=label2,linestyle='None',capsize=5)
    plt.xlabel(r'$\mathrm{BJD }$')
    plt.ylabel(r'$\mathrm{target \ rel. \ flux}$')
    plt.grid(True)
    x_lim = ax1.get_xlim()
    y_lim = ax1.get_ylim()
    plt.legend()

    fig = plt.gcf() # get current figure
    fig.set_size_inches(common_width,4)

    # Plot name
    # Base names:
    base_ = 'binned_vs_filtered'
    # Complete file names:
    plot_name = write.file_name(prefix_,base_,suffix_) + extension
    fig.savefig(plot_name, dpi=400, bbox_inches='tight')
    end = 'Plot (comparison filtered vs. filtered + binned data) saved as:\n' + plot_name + '\n'
    print(end)
    
if BINNED_PERIODOGRAM == True:
    # LS PERIODOGRAM COMPARISON: BINNED DATA VS. RAW / FILTERED DATA
    MT1 = ss.new_flux(MDATA[2],MDATA[5],opt_config)[0][0]
    DMT1 = ss.new_flux_err(MDATA[2],MDATA[3],MDATA[5],MDATA[6],opt_config)[0][0]
    MBJD = MDATA[0][0]
    BJD  = CDATA[0][0]
    
    binned_fluxes = fd.bin_photometry(MDATA,bin_size,opt_config)
    binned_target_flux     = binned_fluxes[1][0]
    binned_target_flux_err = binned_fluxes[2][0]
    
    # NaN values masks corresponding to target from binned_fluxes. 
    masks = binned_fluxes[5]
    t_total_nans = masks[0][0] + masks[1][0]
    
    # Set total mask and obtain target
    # rel. flux corresponding to selected binning with no Nan values
    # for forecoming computations. 
    final_mask = t_total_nans
    size = len(final_mask) - np.sum(final_mask)
    
    BT1        = np.ma.masked_array(data=binned_target_flux,mask=final_mask).compressed()
    BDT1       = np.ma.masked_array(data=binned_target_flux_err,mask=final_mask).compressed()
    BBJD       = np.ma.masked_array(data=binned_fluxes[0][0],mask=final_mask).compressed()
    Bfrequency = np.arange(min_freq_factor * 1. / ( BJD.max() -  BJD.min()), max_freq,1. / (n0 * ( BJD.max() -  BJD.min())))
    Mfrequency = np.arange(min_freq_factor * 1. / ( BJD.max() -  BJD.min()), max_freq,1. / (n0 * ( BJD.max() -  BJD.min())))
    # We consider LS periodogram without error for the binned data, given that
    # the errors tend to be higher in this case, and may produce power maxima
    # at unreasonably short periods. 
    Bpower = LombScargle(BBJD,BT1,BDT1,normalization='standard',nterms=period_terms).power(Bfrequency)
    Mpower = LombScargle(MBJD,MT1,DMT1,normalization='standard',nterms=period_terms).power(Mfrequency)
    BLS    = LombScargle(BBJD,BT1,BDT1,normalization='standard',nterms=period_terms)
    MLS    = LombScargle(MBJD,MT1,DMT1,normalization='standard',nterms=period_terms)

    Bnu_1     = Bfrequency[np.argmax(Bpower)]
    Mnu_1     = Mfrequency[np.argmax(Mpower)]
    BPeriod_1 = 1. / Bfrequency[np.argmax(Bpower)]
    MPeriod_1 = 1. / Mfrequency[np.argmax(Mpower)]
    
    Bt_fit = np.linspace(BBJD.min(),BBJD.max(),10000)
    Mt_fit = np.linspace(MBJD.min(),MBJD.max(),10000)
    By_fit = BLS.model(Bt_fit,Bnu_1)
    My_fit = MLS.model(Mt_fit,Mnu_1)
    
    print("Max. freq.:")
    print("Filtered \t %.8f; Binned \t %.8f" % (Mnu_1,Bnu_1))
    print("Max. powers:")
    print("Filtered \t %.8f; Binned \t %.8f" % (np.amax(Mpower),np.amax(Bpower)))
    
    ax1a = plt.subplot(211)
    ax1  = plt.subplot(311)
    #ax1.set_ylim([MT1.min() - .1 * (MT1.max() - MT1.min()),MT1.max() + .1 * (MT1.max() - MT1.min())])
    
    label1 = r'$\mathrm{filtered \ data \ (main \ period = %.2f \ d, \ freq. = %.4f \ d}^{-1})$' % ((MPeriod_1,Mnu_1))
    label2 = r'$\mathrm{filtered \ + \ binned \ data \ (main \ period = %.2f \ d, \ freq. = %.4f \ d}^{-1})$' % ((BPeriod_1,Bnu_1))

    plt.axvline(x=1./Mnu_1,color='gray',linestyle='--',linewidth=0.75)
    plt.axvline(x=1./Bnu_1,color='darkblue',linestyle='--',linewidth=0.75)
    for i in range(2,4):
        plt.axvline(x=(Bnu_1 / float(i)),color='black',linestyle='--',linewidth=1.)
    max_period = np.asarray([MPeriod_1,BPeriod_1]).max()*2.
    if max_period >= 2. * ( BJD.max() -  BJD.min()): max_period = 2. * ( BJD.max() -  BJD.min())
    ax1.set_xlim([0.,max_freq])
    ax1.plot(Mfrequency,Mpower,linestyle='-',lw=1,marker='o',color='gray',markersize=0.0,label=label1)
    ax1.plot(Bfrequency,Bpower,linestyle='-',lw=1,marker='o',color='darkblue',markersize=0.0,label=label2)
    plt.xlabel(r'$\mathrm{Period \ (days)}$')
    plt.ylabel(r'$\mathrm{Power }$')
    plt.grid(True)
    plt.legend()
    
    ax2 = plt.subplot(312)
    label1 = r'$\mathrm{Lomb-Scargle \ model \ best-fit \ at \ }T_{\mathrm{peak}} = %.2f \ days}$' % (MPeriod_1)
    label2 = r'$\mathrm{filtered \ data}$'
    #Mt_fit = np.linspace(MBJD.min(),MBJD.max(),5000) 
    #My_fit = MLS.model(Mt_fit,Mnu_1)
    #Mphase = fd.fold(MBJD,MPeriod_1)
    #phase_fit = fd.fold(t_fit,MPeriod_1)
    ax2.plot(MBJD,MT1,marker='o',markersize=dot_size,label=label2,color='gray',linestyle='None')
    ax2.plot(Mt_fit,My_fit,color='black',linestyle='-',lw=0.75,label = label1)
    plt.xlabel(r'$\mathrm{BJD }$')
    plt.ylabel(r'$\mathrm{target \ rel. \ flux}$')
    plt.grid(True)
    plt.legend(loc='lower right')
    
    ax2 = plt.subplot(313)
    label1 = r'$\mathrm{Lomb-Scargle \ model \ best-fit \ at \ }T_{\mathrm{peak}} = %.2f}$' % (BPeriod_1)
    label2 = r'$\mathrm{filtered \ + \ binned \ data}$'
    #Bt_fit = np.linspace(BBJD.min(),BBJD.max(),5000) 
    #By_fit = MLS.model(Bt_fit,Bnu_1)
    #Mphase = fd.fold(MBJD,MPeriod_1)
    #phase_fit = fd.fold(t_fit,MPeriod_1)
    ax2.errorbar(BBJD,BT1,BDT1,marker='o',markersize=dot_size,label=label2,color='darkblue',linestyle='None',capsize=5)
    ax2.plot(Bt_fit,By_fit,color='black',linestyle='-',lw=0.75,label = label1)
    plt.xlabel(r'$\mathrm{BJD }$')
    plt.ylabel(r'$\mathrm{target \ rel. \ flux}$')
    plt.grid(True)
    plt.legend(loc='lower right')

    fig = plt.gcf() # get current figure
    s = 1.3
    fig.set_size_inches(s*common_width,s*3.25*3)
    # Base name:
    base_ = 'lombscargle_binned_vs_filtered'
    # Complete file names:
    plot_name = write.file_name(prefix_,base_,suffix_) + extension
    fig.savefig(plot_name, dpi=400, bbox_inches='tight')
    end = 'Plot (compare Lomb-Scargle filtered vs. binned + filtered data) saved as:\n' + plot_name + '\n'
    print(end)
    
if BINNED_PHASE_FOLDING == True:
    # Phase folding of time series for binned filtered flux data
    MT1 = ss.new_flux(MDATA[2],MDATA[5],opt_config)[0][0]
    DMT1 = ss.new_flux_err(MDATA[2],MDATA[3],MDATA[5],MDATA[6],opt_config)[0][0]
    MBJD = MDATA[0][0]
    BJD  = CDATA[0][0]

    binned_fluxes = fd.bin_photometry(MDATA,bin_size,opt_config)
    binned_target_flux     = binned_fluxes[1][0]
    binned_target_flux_err = binned_fluxes[2][0]
    
    # NaN values masks corresponding to target from binned_fluxes. 
    masks = binned_fluxes[5]
    t_total_nans = masks[0][0] + masks[1][0]
    
    # Set total mask and obtain target
    # rel. flux corresponding to selected binning with no Nan values
    # for forecoming computations. 
    final_mask = t_total_nans
    size = len(final_mask) - np.sum(final_mask)
    
    BT1        = np.ma.masked_array(data=binned_target_flux,mask=final_mask).compressed()
    BDT1       = np.ma.masked_array(data=binned_target_flux_err,mask=final_mask).compressed()
    BBJD       = np.ma.masked_array(data=binned_fluxes[0][0],mask=final_mask).compressed()
    Bfrequency = np.arange(min_freq_factor * 1. / (BBJD.max() - BBJD.min()), max_freq,1. / (n0 * (BBJD.max() - BBJD.min())))
    Mfrequency = np.arange(min_freq_factor * 1. / (MBJD.max() - MBJD.min()), max_freq,1. / (n0 * (MBJD.max() - MBJD.min())))
    # We consider LS periodogram without error for the binned data, given that
    # the errors tend to be higher in this case, and may produce power maxima
    # at unreasonably short periods. 
    Bpower = LombScargle(BBJD,BT1,BDT1,normalization='standard',nterms=period_terms).power(Bfrequency)
    Mpower = LombScargle(MBJD,MT1,DMT1,normalization='standard',nterms=period_terms).power(Mfrequency)
    BLS    = LombScargle(BBJD,BT1,BDT1,normalization='standard',nterms=period_terms)
    MLS    = LombScargle(MBJD,MT1,DMT1,normalization='standard',nterms=period_terms)

    Bnu_1     = Bfrequency[np.argmax(Bpower)]
    Mnu_1     = Mfrequency[np.argmax(Mpower)]
    
    BPeriod_1 = 1. / Bnu_1
    MPeriod_1 = 1. / Mnu_1
    
    Bt_fit = np.linspace(BBJD.min(),BBJD.max(),10000)
    Mt_fit = np.linspace(MBJD.min(),MBJD.max(),10000)
    By_fit = BLS.model(Bt_fit,Bnu_1)
    My_fit = MLS.model(Mt_fit,Mnu_1)
    
    number_of_periods = 1. # has to be an integer
   
    fold_MBJD    = number_of_periods * fd.fold(MBJD,number_of_periods * MPeriod_1)
    fold_BBJD   = number_of_periods * fd.fold(BBJD,number_of_periods * BPeriod_1)
    
    fold_Mt_fit = number_of_periods * fd.fold(Mt_fit,number_of_periods * MPeriod_1)
    fold_Bt_fit = number_of_periods * fd.fold(Bt_fit,number_of_periods * BPeriod_1)
    
    ax1 = plt.subplot(211)
    ax1.set_ylim([My_fit.min() - 2. * (My_fit.max() - My_fit.min()),My_fit.max() + 2. * (My_fit.max() - My_fit.min())])
    label11 = r'$\mathrm{filtered \ data}$'
    label12 = r'$\mathrm{Lomb-Scargle \ model \ best-fit \ at \ }T_{\mathrm{peak}} = %.2f$' % (1./Mnu_1)
    label21 = r'$\mathrm{filtered \  + \ binned \ data}$' 
    label22 = r'$\mathrm{Lomb-Scargle \ model \ best-fit \ at \ }T_{\mathrm{peak}} = %.2f$' % (1./Bnu_1)
    ax1.plot(fold_MBJD,MT1,marker='o',color='gray',markersize=dot_size*1.25,label=label11,linestyle='None')
    ax1.plot(fold_Mt_fit,My_fit,color='black',label=label12,marker='o',markersize=dot_size / 2.,linestyle='None')
    #ax1.errorbar(BJD,T1,yerr=DATA[3][0,:]*1.,marker='o',color='red',markersize=1,label=label1,linestyle='None',capsize=5)
    plt.xlabel(r'$\mathrm{Phase} / 2\pi$')
    plt.ylabel(r'$\mathrm{target \ rel. \ flux}$')
    plt.grid(True)
    plt.legend(loc='lower right')

    ax2 = plt.subplot(212)
    ax2.set_ylim([By_fit.min() - 2. * (By_fit.max() - By_fit.min()),By_fit.max() + 2. * (By_fit.max() - By_fit.min())])
    # ax.set_xlim([2458935,2459002])
    #ax1.errorbar(BJD,T1,yerr=DATA[3][0,:]*1.,marker='o',color='red',markersize=1,label=label1,linestyle='None',capsize=5)
    ax2.errorbar(fold_BBJD,BT1,yerr=BDT1,marker='o',markersize=dot_size*1.25,label=label21,color='darkblue',linestyle='None',capsize = 5)
    ax2.plot(fold_Bt_fit,By_fit,color='black',label=label22,marker='o',markersize=dot_size / 2.,linestyle='None')
    plt.xlabel(r'$\mathrm{Phase} / 2\pi$')
    plt.ylabel(r'$\mathrm{target \ rel. \ flux}$')
    plt.grid(True)
    plt.legend(loc='lower right')
    s = 1.5
    fig = plt.gcf() # get current figure
    fig.set_size_inches(s*common_width,s*3.3*2)
    
    # Base names:
    base_ = 'phase_folding_binned'
    # Complete file names:
    plot_name = write.file_name(prefix_,base_,suffix_) + extension
    fig.savefig(plot_name, dpi=400, bbox_inches='tight')
    end = 'Plot (phase folding binned) saved as:\n' + plot_name + '\n'
    print(end)

if RMS_VS_SELECTED == True:
    # RMS VS. SELECTED COMP. STARS
    '''
    # Filtered data
    with open(filtered_data_filename) as f:
        lines = f.readlines()
        x1 = [float(line.split()[0]) for line in lines]
        y1 = [float(line.split()[1]) for line in lines]
    '''
    
    s = 1.1
    
    params = {'legend.fontsize': 12*s,
            'axes.labelsize': 14*s,
            'axes.titlesize': 12*s,
            'xtick.labelsize': 10*s,
            'ytick.labelsize': 10*s}
    pylab.rcParams.update(params)
    
    # Raw data
    with open(raw_data_filename1) as g1:
        lines = g1.readlines()
        xr1 = [float(line.split()[0]) for line in lines]
        yr1 = [float(line.split()[1]) for line in lines]
    with open(raw_data_filename2) as g2:
        lines = g2.readlines()
        xr2 = [float(line.split()[0]) for line in lines]
        yr2 = [float(line.split()[1]) for line in lines]
    with open(raw_data_filename3) as g3:
        lines = g3.readlines()
        xr3 = [float(line.split()[0]) for line in lines]
        yr3 = [float(line.split()[1]) for line in lines]
    with open(raw_data_filename4) as g4:
        lines = g4.readlines()
        xr4 = [float(line.split()[0]) for line in lines]
        yr4 = [float(line.split()[1]) for line in lines]
    
    xr1_rev = [0. for i in range(0,len(xr1))]
    xr2_rev = [0. for i in range(0,len(xr2))]
    xr3_rev = [0. for i in range(0,len(xr3))]
    xr4_rev = [0. for i in range(0,len(xr4))]
    for j in range(0,len(xr1)): xr1_rev[j] = comp_stars1 - xr1[j]
    for j in range(0,len(xr2)): xr2_rev[j] = comp_stars2 - xr2[j]
    for j in range(0,len(xr3)): xr3_rev[j] = comp_stars3 - xr3[j]
    for j in range(0,len(xr4)): xr4_rev[j] = comp_stars4 - xr4[j]
    '''
    # Binned data
    with open(binned_data_filename) as z:
        lines = z.readlines()
        x3 = [float(line.split()[0]) for line in lines]
        y3 = [float(line.split()[1]) for line in lines]
    '''
    label1 = r'$\mathrm{TZ \ Ari}$'
    label2 = r'$\mathrm{GJ \ 555}$'
    label3 = r'$\mathrm{TOI-1266}$'
    label4 = r'$\mathrm{Wolf \ 1069}$'
    
    #label1 =  r'$\mathrm{filtered \ data} \ (\pm %.2f \sigma, p=%.2f)$' % ((t_su,t_p))
    #label3 =  r'$\mathrm{filtered \ + \ binned \ data} \ (\pm %.2f \sigma, p=%.2f, bin size = %.2f h)$' % ((t_su,t_p,bin_size*24.))
    #label2 =  r'$\mathrm{raw \ data}$'  
    #x1_rev = [0. for i in range(0,len(x1))]
    #x2_rev = [0. for i in range(0,len(x2))]
    #x3_rev = [0. for i in range(0,len(x3))]
    #for i in range(0,len(x1)): x1_rev[i] = float(comp_stars) - x1[i]
    #for j in range(0,len(x2)): x2_rev[j] = float(comp_stars) - x2[j]
    #for i in range(0,len(x3)): x3_rev[i] = float(comp_stars) - x3[i]
    #rmss = np.asarray([yr1,yr2,yr3,yr4])
    #ax1.set_ylim([rmss.min() - .1 * (rmss.max() - rmss.min()),rmss.max() + .1 * (rmss.max() - rmss.min())])

    fig, axs = plt.subplots(2,2,sharex=False)
    # ax.set_xlim([2458935,2459002])
    #plt.axhline(y=mean_T1,color='r',linestyle='-',linewidth=1)
    #plt.grid(True)
    #iqr_value = '$\mathrm{IQR} = %.4f$' % ((IQRs[1][0]))
    #anchored_text = AnchoredText(iqr_value, loc="lower right",frameon=False)
    #axs[i,0].add_artist(anchored_text)

    raw_min1      = np.argmin(yr1)
    raw_min2      = np.argmin(yr2)
    raw_min3      = np.argmin(yr3)
    raw_min4      = np.argmin(yr4)
    
    color1 = 'steelblue'
    color2 = 'cornflowerblue'
    color3 = 'tomato'
    color4 = 'lightsalmon'
    axs[0,0].plot(xr1_rev,np.asarray(yr1),marker='o',markersize=dot_size * 2, lw=1,color = color1,label=label1)

    #plt.yticks(np.arange(0., 1.25, step=0.25),[0,'',0.5,'',1.])
    #axs[0,0].yaxis.tick_right()
    axs[0,0].plot(xr1_rev[raw_min1],yr1[raw_min1],marker='o',markersize = dot_size*2.5,color='black')
    yr1min = np.asarray(yr1).min()
    yr1max = np.asarray(yr1).max()
    yr2min = np.asarray(yr2).min()
    yr2max = np.asarray(yr2).max()
    yr3min = np.asarray(yr3).min()
    yr3max = np.asarray(yr3).max()
    yr4min = np.asarray(yr4).min()
    yr4max = np.asarray(yr4).max()
    steps  = 6.
    c_ = 0.25
    y1space = (yr1max - yr1min) / (steps - 1.)
    axs[0,0].set_yticks(np.arange(yr1min - y1space*c_,yr1max + y1space*c_,y1space))
    
    y2space = (yr2max - yr2min) / (steps - 1.)
    axs[0,1].set_yticks(np.arange(yr2min - y2space*c_,yr2max + y2space*c_,y2space))
    
    y3space = (yr3max - yr3min) / (steps - 1.)
    axs[1,0].set_yticks(np.arange(yr3min - y3space*c_,yr3max + y3space*c_,y3space))
    
    y4space = (yr4max - yr4min) / (steps - 1.)
    axs[1,1].set_yticks(np.arange(yr4min - y4space*c_,yr4max + y4space*c_,y4space))
    
    axs[0,1].plot(xr2_rev,np.asarray(yr2),marker='o',markersize=dot_size * 2, lw=1,color=color2,label=label2)
    axs[0,1].legend()

    axs[0,1].plot(xr2_rev[raw_min2],yr2[raw_min2],marker='o',markersize = dot_size*2.5,color='black') 
    axs[1,0].plot(xr3_rev,np.asarray(yr3),marker='o',markersize=dot_size * 2, lw=1,color=color3,label=label3)
    axs[1,0].legend()
    axs[1,0].plot(xr3_rev[raw_min3],yr3[raw_min3],marker='o',markersize = dot_size*2.5,color='black')
    axs[1,1].plot(xr4_rev,np.asarray(yr4),marker='o',markersize=dot_size * 2, lw=1,color=color4,label=label4)
    

    #ax1.plot(x3_rev,y3,marker='o',markersize=dot_size * 2, lw=1,color='darkblue',label=label2)
    #filtered_min  = np.argmin(y1)
    #binned_min    = np.argmin(y3)

    axs[1,1].plot(xr4_rev[raw_min4],yr4[raw_min4],marker='o',markersize = dot_size*2.5,color='black') 
    
    for i in range(0,2):
        for j in range(0,2):
            if i != 0: axs[i,j].set_xlabel(r"$\mathrm{\# \ selected \ comp. \ stars}$")
            if j != 1: axs[i,j].set_ylabel(r"$\mathbf{\sigma}_{\mathrm{min}}$")
            
    axs[0,0].set_xticks([i for i in range(0,15)])
    axs[0,1].set_xticks([i for i in range(0,15)])
    axs[1,0].set_xticks([i for i in range(0,15)])
    axs[1,1].set_xticks([i for i in range(0,15)])
    
    axs[0,0].legend()
    axs[0,1].legend()
    axs[1,0].legend()
    axs[1,1].legend()
   
    axs[0,0].grid(True)
    axs[1,0].grid(True)
    axs[0,1].grid(True)
    axs[1,1].grid(True)
    
    axs[0,1].yaxis.tick_right()
    axs[1,1].yaxis.tick_right()
    #fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    #plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    #plt.ylabel(r'$\mathrm{(selected) \ comp. \ stars \ norm. \ rel. \ flux}$',labelpad=15.)
    #fig.subplots_adjust(hspace=0.,wspace = 0.)
    
    #plt.legend()
    fig1 = plt.gcf() # get current figure
    # 10 comp. star -> size = 12 x 16.
    # height of 1 comp. star is 16 / 10 = 1.6. 
    k = 2
    fig1.set_size_inches(common_width*s,3. * s * k)
    plt.tight_layout()
    
    #plt.plot(x2_rev[raw_min],y2[raw_min],marker='o',markersize = dot_size * 2.5,color='black') 
    #plt.plot(x3_rev[binned_min],y3[binned_min],marker='o',markersize = dot_size*2.5,color='black') 
    #plt.xlabel(r"$\mathrm{\# \ selected \ comp. \ stars}$")
    #plt.ylabel(r"$\mathbf{\sigma}_{\mathrm{min}}$")
    #plt.grid(True)
    #plt.xticks(np.arange(0, comp_stars, step=1))
    #plt.legend()
    # Base names:
    base_ = 'rms_vs_selected_raw_general'
    # Complete file names:
    plot_name = write.file_name(prefix_,base_,suffix_) + extension
    fig1.savefig(plot_name, dpi=400, bbox_inches='tight')
    end = 'Plot (min. rms vs. selected comp. stars) saved as:\n' + plot_name + '\n'
    print(end)

if RMS_VS_BINSIZE == True:
    # RMS VS BIN SIZE
    # Filtered data
    with open(filename) as f:
        lines = f.readlines()
        x1 = [float(line.split()[1]) for line in lines]
        y1 = [float(line.split()[2]) for line in lines]
    
    ax1a = plt.subplot(211)
    ax1  = plt.subplot(111)
    ax1.plot(np.sqrt(np.asarray(x1)),np.asarray(y1),marker='o',markersize=dot_size*1.5, linestyle='none',color='black')
    plt.axvline(x=math.sqrt(bin_left),color='gray',linestyle='--',linewidth=0.75)
    plt.axvline(x=math.sqrt(bin_right),color='gray',linestyle='--',linewidth=0.75)
    plt.axvline(x=math.sqrt(x1[    int(np.argmin(np.asarray(rms_sub_1)))+int(j_bin)]),color='darkred',linestyle='--',linewidth=0.75)
    plt.xlabel(r"$\sqrt{\mathrm{bin \ size}}$")
    plt.ylabel(r"$\mathrm{target \ rel. \ flux \ rms}$")
    plt.grid(True)

    #plt.xticks(np.arange(0, 15, step=1))
    #plt.legend()
    fig = plt.gcf() # get current figure
    fig.set_size_inches(common_width,4)
    # Base names:
    base_ = 'rms_vs_bin_size'
    # Complete file names:
    plot_name = write.file_name(prefix_,base_,suffix_) + extension
    fig.savefig(plot_name, dpi=400, bbox_inches='tight')
    end = 'Plot (rms vs. bin size) saved as:\n' + plot_name + '\n'
    print(end)


if RAW_FILTERED_COMPARISON == True:

    s = 1.25
    params = {'legend.fontsize': 10*s,
            'axes.labelsize': 12*s,
            'axes.titlesize': 12*s,
            'xtick.labelsize': 10*s,
            'ytick.labelsize': 10*s}
    pylab.rcParams.update(params)
    dot_size = 2.15*s
    # Raw data vs. filtered data comparison
    # Get non-discarded data from previous filtering.
    T1  = CDATA[2][0]
    DT1 = CDATA[3][0]
    MT1 = ss.new_flux(MDATA[2],MDATA[5],opt_config)[0][0]
    DMT1 = ss.new_flux_err(MDATA[2],MDATA[3],MDATA[5],MDATA[6],opt_config)[0][0]
    BJD = CDATA[0][0]
    MBJD = MDATA[0][0]

    MT1_copy = MT1
    T1_copy = T1
    Mrms = vi.rms(MT1,DMT1,weighted)
    Rrms = vi.rms(T1,DT1,weighted)

    dev_T1  = np.std(T1,dtype=np.float64)
    mean_T1 = np.mean(T1,dtype=np.float64)
    
    Rnorm = mean_T1
    dev_T1 /=  Rnorm
    mean_T1 /= Rnorm

    dev_MT1  = np.std(MT1,dtype=np.float64)
    mean_MT1 = np.mean(MT1,dtype=np.float64)
    Mnorm = mean_MT1
    dev_MT1 /=  Mnorm
    mean_MT1 /= Mnorm
    T1_copy = T1 / Rnorm
    MT1_copy = MT1 / Mnorm
    
    ax1 = plt.subplot(411)
    ax1.set_ylim([mean_T1-3.*dev_T1,mean_T1+3.*dev_T1])
    plt.axhline(y=mean_T1,color='r',linestyle='-',linewidth=1)
    plt.axhline(y=mean_T1-dev_T1,color='r',linestyle='--',linewidth=1)
    plt.axhline(y=mean_T1+dev_T1,color='r',linestyle='--',linewidth=1)
    plt.axhline(y=mean_MT1,color='gray',linestyle='-',linewidth=1)
    plt.axhline(y=mean_MT1-dev_MT1,color='gray',linestyle='--',linewidth=1)
    plt.axhline(y=mean_MT1+dev_MT1,color='gray',linestyle='--',linewidth=1)
    label1 = r'$\mathrm{raw \ data}$' 
    label2 = r'$\mathrm{filtered \ data} \ $' 
    ax1.plot(BJD,T1_copy,marker='o',color='red',markersize=dot_size,label=label1,linestyle='None')
    #ax1.errorbar(BJD,T1,yerr=DATA[3][0,:]*1.,marker='o',color='red',markersize=1,label=label1,linestyle='None',capsize=5)
    ax1.plot(MBJD,MT1_copy,marker='o',markersize=dot_size,label=label2,color='gray',linestyle='None')
    plt.xlabel(r'$\mathrm{BJD }$')
    plt.ylabel(r'$\mathrm{norm. target \ rel. \ flux}$')
    plt.grid(True)
    plt.legend(loc='lower right')

    ax2 = plt.subplot(412)
    SNR_T1 = CDATA[4][0]
    margin_DATA = fd.mask_data(CDATA,fd.margin(SNR_T1,t_p).mask)
    margin_SNR_T1 = margin_DATA[4][0]
    margin_BJD    = margin_DATA[0][0]
    #ax2.set_ylim([0,SNR_T1.max()*(1 + 0.15)])
    plt.axhline(y=t_p*SNR_T1.max(),color='darkgreen',linestyle='--',linewidth=1)
    label1 = r'$\mathrm{raw \ data}$' 
    label2 = r'$\mathrm{filtered \ data} \ $'
    ax2.plot(BJD,SNR_T1,marker='o',color='darkgreen',markersize=dot_size,label=label1,linestyle='None')
    ax2.plot(margin_BJD,margin_SNR_T1,marker='o',markersize=dot_size,label=label2,color='gray',linestyle='None')
    plt.xlabel(r'$\mathrm{BJD }$')
    plt.ylabel(r'$\mathrm{target \ SNR}$')
    plt.grid(True)
    plt.legend(loc='lower right')
    
    ax3 = plt.subplot(413)
    DT1 = CDATA[3][0]
    DMDATA = ss.new_flux_err(MDATA[2],MDATA[3],MDATA[5],MDATA[6],opt_config)
    DMT1 = DMDATA[0][0]
    frequency  = np.arange(min_freq_factor * 1. / ( BJD.max() -  BJD.min()), max_freq ,1. / (n0 * ( BJD.max() -  BJD.min())))
    Mfrequency = np.arange(min_freq_factor * 1. / ( BJD.max() -  BJD.min()), max_freq ,1. / (n0 * ( BJD.max() -  BJD.min())))
    power  = LombScargle( BJD, T1, DT1,normalization='standard',nterms=period_terms).power(frequency)
    Mpower = LombScargle(MBJD,MT1,DMT1,normalization='standard',nterms=period_terms).power(Mfrequency)
    MLS    = LombScargle(MBJD,MT1,DMT1,normalization='standard',nterms=period_terms)
    RLS    = LombScargle( BJD, T1, DT1,normalization='standard',nterms=period_terms)
    nu_1      =  frequency[np.argmax(power )]
    Mnu_1     = Mfrequency[np.argmax(Mpower)]
    Period_1  = 1. /  frequency[np.argmax(power )]
    MPeriod_1 = 1. / Mfrequency[np.argmax(Mpower)]
    label1 = r'$\mathrm{raw \ data \ (main \ period = %.2f \ d, \ freq. = %.4f \ d}^{-1})$' % ((Period_1,nu_1))
    label2 = r'$\mathrm{filtered \ data} \   \ \mathrm{(main \ period = %.2f \ d, \ freq. = %.4f \ d}^{-1})$' % ((MPeriod_1,Mnu_1))
    plt.axvline(x=nu_1,color='blue',linestyle='--',linewidth=1.)
    plt.axvline(x=Mnu_1,color='gray',linestyle='--',linewidth=1.)
    faps_probs = [0.1/100.,0.01/100.,0.001/100.]
    Rlevels = RLS.false_alarm_level(faps_probs)  
    Mlevels = MLS.false_alarm_level(faps_probs)
    plt.axhline(y=Mlevels[0],color='black',linestyle='dotted',linewidth=1.4,)
    plt.axhline(y=Mlevels[1],color='black',linestyle='dashdot',linewidth=1.4)
    plt.axhline(y=Mlevels[2],color='black',linestyle='solid',linewidth=1.4)
    #for i in range(2,4):
    #    plt.axvline(x=Mnu_1 / float(i),color='black',linestyle='--',linewidth=1.)
    ax3.set_xlim([0.,max_freq])
    ax3.plot(frequency,power,linestyle='-',lw=1,marker='o',color='blue',markersize=0.0,label=label1)
    ax3.plot(Mfrequency,Mpower,linestyle='-',lw=1,marker='o',color='gray',markersize=0.0,label=label2)
    plt.xlabel(r'$\nu \ (\mathrm{d}^{-1})$')
    plt.ylabel(r'$\mathrm{Power}$')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    ax4 = plt.subplot(414)
    label1 = r'$\mathrm{LS \ model \ best-fit \ at \ }T_{\mathrm{peak}} = %.2f \ \mathrm{days}$' % (MPeriod_1)
    label2 = r'$\mathrm{filtered \ data} \  $'
    Mt_fit = np.linspace(MBJD.min(),MBJD.max(),5000) 
    My_fit = MLS.model(Mt_fit,Mnu_1)
    theta = MLS.model_parameters(Mfrequency[np.argmax(Mpower)])
    offset = MLS.offset()
    design_matrix = MLS.design_matrix(Mnu_1,Mt_fit)
    modelys = offset + design_matrix.dot(theta)
    ax4.plot(MBJD,MT1,marker='o',markersize=dot_size,label=label2,color='gray',linestyle='None')
    #ax4.plot(MBJD,MT1,marker='o',markersize=dot_size,label=label2,color='blue',linestyle='None')
    ax4.plot(Mt_fit,modelys,color='black',linestyle='-',lw=1.*s,label = label1)
    plt.xlabel(r'$\mathrm{BJD }$')
    plt.ylabel(r'$\mathrm{target \ rel. \ flux}$')
    plt.grid(True)
    plt.legend(loc='lower right')

    plt.tight_layout()
    fig = plt.gcf() # get current figure
    fig.set_size_inches(s*common_width*1.,s*2.5*4)
    plt.tight_layout()
    # Base name:
    base_ = 'flux_snr_lombscargle'
    # Complete file names:
    plot_name = write.file_name(prefix_,base_,suffix_) + extension
    fig.savefig(plot_name, dpi=400, bbox_inches='tight')
    
    print("Max. freq.:")
    print("Raw \t %.8f; Filtered \t %.8f" % (nu_1,Mnu_1))
    print("Max. powers:")
    print("Raw \t %.8f; Filtered \t %.8f" % (np.amax(power),np.amax(Mpower)))
    print("False Alarm Probabilities at peak powers:")
    print("Raw data:\t",RLS.false_alarm_probability(power.max()))
    print("Filtered data:\t",MLS.false_alarm_probability(Mpower.max()))
    print("\n")
    print("FAP levels:")
    print("Raw data:\t",Rlevels)
    print("Filtered data:\t",Mlevels)
    print("Lomb-Scargle filtered data best-fit parameters:")
    print(theta)
    print("\n")
    end = 'Plot (compare raw vs. filtered data) saved as:\n' + plot_name + '\n'
    print(end)


if PHASE_FOLDING == True:
    
    s = 1.25

    params = {'legend.fontsize': 10*s,
        'axes.labelsize': 12*s,
        'axes.titlesize': 12*s,
        'xtick.labelsize': 10*s,
        'ytick.labelsize': 10*s}
    pylab.rcParams.update(params)

    dot_size = 2.15*s
    # Phase folding of time series
    T1  = CDATA[2][0]
    MT1 = ss.new_flux(MDATA[2],MDATA[5],opt_config)[0][0]
    BJD = CDATA[0][0]
    MBJD = MDATA[0][0]
    DT1 = CDATA[3][0]
    DMDATA = ss.new_flux_err(MDATA[2],MDATA[3],MDATA[5],MDATA[6],opt_config)
    DMT1 = DMDATA[0][0]
    
    frequency  = np.arange(min_freq_factor * 1. / ( BJD.max() -  BJD.min()), max_freq,1. / (n0 * ( BJD.max() -  BJD.min())))
    Mfrequency = np.arange(min_freq_factor * 1. / ( BJD.max() -  BJD.min()), max_freq,1. / (n0 * ( BJD.max() -  BJD.min())))

    power  = LombScargle( BJD, T1, DT1,normalization='standard',nterms=period_terms).power(frequency)
    Mpower = LombScargle(MBJD,MT1,DMT1,normalization='standard',nterms=period_terms).power(Mfrequency)
    MLS    = LombScargle(MBJD,MT1,DMT1,normalization='standard',nterms=period_terms)
    RLS    = LombScargle( BJD, T1, DT1,normalization='standard',nterms=period_terms)

    nu_1      =  frequency[np.argmax(power )]
    Mnu_1     = Mfrequency[np.argmax(Mpower)]
    
    number_of_periods = 1. # has to be an integer
    Rt_fit = np.linspace(BJD.min(),BJD.max(),10000)
    Mt_fit = np.linspace(MBJD.min(),MBJD.max(),10000)
    Ry_fit = RLS.model(Rt_fit,Mnu_1)
    My_fit = MLS.model(Mt_fit,Mnu_1)
    Period_1 = 1. / Mnu_1
    MPeriod_1 = 1. / Mnu_1

    fold_BJD    = number_of_periods * fd.fold(BJD,number_of_periods * Period_1)
    fold_MBJD   = number_of_periods * fd.fold(MBJD,number_of_periods * MPeriod_1)
    
    fold_Rt_fit = number_of_periods * fd.fold(Rt_fit,number_of_periods * Period_1)
    fold_Mt_fit = number_of_periods * fd.fold(Mt_fit,number_of_periods * MPeriod_1)
    
    ax1 = plt.subplot(211)
    ax1.set_ylim([Ry_fit.min() - amp* (Ry_fit.max() - Ry_fit.min()),Ry_fit.max() + amp * (Ry_fit.max() - Ry_fit.min())])
    label11 = r'$\mathrm{raw \ data}$'
    label12 = r'$\mathrm{LS \ model \ best-fit \ at \ }T_{\mathrm{peak}} = %.2f \ \mathrm{days}$' % (1./nu_1)
    label21 = r'$\mathrm{filtered \ (shifted) \ data} \  $' 
    label22 = r'$\mathrm{LS \ model \ best-fit \ at \ }T_{\mathrm{peak}} = %.2f \ \mathrm{days}$' % (1./Mnu_1)
    ax1.plot(fold_BJD,T1,marker='o',color='red',markersize=dot_size*1.25,label=label11,linestyle='None')
    ax1.plot(fold_Rt_fit,Ry_fit,color='black',label=label12,marker='o',markersize=dot_size / 2.,linestyle='None')
    #ax1.errorbar(BJD,T1,yerr=DATA[3][0,:]*1.,marker='o',color='red',markersize=1,label=label1,linestyle='None',capsize=5)
    plt.xlabel(r'$\mathrm{Phase} / 2\pi$')
    plt.ylabel(r'$\mathrm{target \ rel. \ flux}$')
    plt.grid(True)
    plt.legend(loc='upper left')
    '''
    ax2 = plt.subplot(223)
    ax2.set_ylim([My_fit.min() - amp * (My_fit.max() - My_fit.min()),My_fit.max() + amp * (My_fit.max() - My_fit.min())])
    ax2.plot(fold_MBJD,MT1,marker='o',markersize=dot_size*1.25,label=label21,color='lightsalmon',linestyle='None')
    ax2.plot(fold_Mt_fit,My_fit,color='black',label=label22,marker='o',markersize=dot_size / 2.,linestyle='None')
    plt.xlabel(r'$\mathrm{Phase} / 2\pi$')
    plt.ylabel(r'$\mathrm{target \ (shifted) \ rel. \ flux}$')
    plt.grid(True)
    plt.legend(loc='upper left')
    '''
    
    number_of_periods = 1. # has to be an integer
    Rt_fit = np.linspace(BJD.min(),BJD.max(),10000)
    Mt_fit = np.linspace(MBJD.min(),MBJD.max(),10000)
    Ry_fit = RLS.model(Rt_fit,nu_1)
    My_fit = MLS.model(Mt_fit,Mnu_1)
    Period_1 = 1. / nu_1
    MPeriod_1 = 1. / Mnu_1
    fold_BJD    = number_of_periods * fd.fold(BJD,number_of_periods * Period_1)
    fold_MBJD   = number_of_periods * fd.fold(MBJD,number_of_periods * MPeriod_1)
    
    fold_Rt_fit = number_of_periods * fd.fold(Rt_fit,number_of_periods * Period_1)
    fold_Mt_fit = number_of_periods * fd.fold(Mt_fit,number_of_periods * MPeriod_1)
    '''
    ax3 = plt.subplot(212)
    ax3.set_ylim([Ry_fit.min() - amp* (Ry_fit.max() - Ry_fit.min()),Ry_fit.max() + amp * (Ry_fit.max() - Ry_fit.min())])
    label11 = r'$\mathrm{raw \ data}$'
    label12 = r'$\mathrm{LS \ model \ best-fit \ at \ }T_{\mathrm{peak}} = %.2f \ \mathrm{days}$' % (1./nu_1)
    label21 = r'$\mathrm{filtered \ (shifted) \ data} \  $' 
    label22 = r'$\mathrm{LS \ model \ best-fit \ at \ }T_{\mathrm{peak}} = %.2f \ \mathrm{days}$' % (1./Mnu_1)
    ax3.plot(fold_BJD,T1,marker='o',color='red',markersize=dot_size*1.25,label=label11,linestyle='None')
    ax3.plot(fold_Rt_fit,Ry_fit,color='black',label=label12,marker='o',markersize=dot_size / 2.,linestyle='None')
    #ax1.errorbar(BJD,T1,yerr=DATA[3][0,:]*1.,marker='o',color='red',markersize=1,label=label1,linestyle='None',capsize=5)
    plt.xlabel(r'$\mathrm{Phase} / 2\pi$')
    plt.ylabel(r'$\mathrm{target \ rel. \ flux}$')
    plt.grid(True)
    plt.legend(loc='upper left')
    '''
    ax4 = plt.subplot(212)
    ax4.set_ylim([My_fit.min() - amp * (My_fit.max() - My_fit.min()),My_fit.max() + amp * (My_fit.max() - My_fit.min())])
    ax4.plot(fold_MBJD,MT1,marker='o',markersize=dot_size*1.25,label=label21,color='lightsalmon',linestyle='None')
    ax4.plot(fold_Mt_fit,My_fit,color='black',label=label22,marker='o',markersize=dot_size / 2.,linestyle='None')
    plt.xlabel(r'$\mathrm{Phase} / 2\pi$')
    plt.ylabel(r'$\mathrm{target \ rel. \ flux}$')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    fig = plt.gcf() # get current figure
    fig.set_size_inches(s*common_width*1.,s*3.5*2.)
    plt.tight_layout()
    
    # Base names:
    base_ = 'phase_folding_combined_'
    # Complete file names:
    plot_name = write.file_name(prefix_,base_,suffix_) + extension
    fig.savefig(plot_name, dpi=400, bbox_inches='tight')
    end = 'Plot (phase folding) saved as:\n' + plot_name + '\n'
    print(end)

if COMP_STAR_VAR_INDICES == True:
    s = 1.25
    params = {'legend.fontsize': 9.*s,
            'axes.labelsize': 12*s,
            'axes.titlesize': 12*s,
            'xtick.labelsize': 10*s,
            'ytick.labelsize': 10*s}
    pylab.rcParams.update(params)
    # Comp. star rel. fluxes and var. indices
    #opt_config= [0,0,0,0,0,0,1,1,1,0,0,0,0]
    config    = np.asarray([1 for _ in range(0,comp_stars)]) - np.asarray(opt_config)
    k = 0
    star_index = []
    max_flux   = []
    dot_size   = 1.5

    BJD  = CDATA[0][0]
    MBJD = MDATA[0][0]
    COMP_FLUXES   = CDATA[5]
    MCOMP_FLUXES  = MDATA[5]
    DMCOMP_FLUXES = MDATA[6]

    # Count how many comp. stars are selected
    # in the current config. 
    # Save max. rel. fluxes of selected comp. stars
    # in max_flux for normalization. 
    for i in range(0,comp_stars):
        if config[i] == 0: 
            k += 1
            star_index.append(i)
            max_flux.append(COMP_FLUXES[i].max())
    
    # Get maximum rel. flux from all selected comp. stars. 
    MAX = np.asarray(max_flux).max()

    # Compute all 4 variability indices for all comp. stars
    rmss     = ss.select_rms(MCOMP_FLUXES,DMCOMP_FLUXES,opts[0],ks[0],thresholds[0])
    IQRs     = ss.select_IQR(MCOMP_FLUXES,opts[1],ks[1],thresholds[1])
    neumanns = ss.select_neumann(MCOMP_FLUXES,opts[2],ks[2],thresholds[2])
    chi2s    = ss.select_chi2(MCOMP_FLUXES,DMCOMP_FLUXES,opts[3],ks[3],thresholds[3])
    print(chi2s[1]/chi2s[1].max())
    # Normalize each index for the selected comp. stars
    # w.r.t. to its maximum value across all comp. stars.  
    yindices = np.zeros((k,4))
    for i in range(0,k):
        j = star_index[i]
        yindices[i][0] = rmss[1][j]     / rmss[1].max()
        yindices[i][1] = IQRs[1][j]     / IQRs[1].max()
        yindices[i][2] = neumanns[1][j] / neumanns[1].max()
        yindices[i][3] = chi2s[1][j]    / chi2s[1].max()
         
    
    # Obtain comp. rel flux w.r.t. to selected ensemble:
    size  = len(COMP_FLUXES[0])
    Msize = len(MCOMP_FLUXES[0])
    copy_COMP_FLUXES  = np.zeros((comp_stars,size))
    copy_MCOMP_FLUXES = np.zeros((comp_stars,Msize))
    for l in range(0,k):
        copy_COMP_FLUXES[star_index[l]] = (COMP_FLUXES[star_index[l]] - MCOMP_FLUXES[star_index[l]].min()) / (MCOMP_FLUXES[star_index[l]].max() -MCOMP_FLUXES[star_index[l]].min())
        copy_MCOMP_FLUXES[star_index[l]] = (MCOMP_FLUXES[star_index[l]] - MCOMP_FLUXES[star_index[l]].min()) / (MCOMP_FLUXES[star_index[l]].max() -  MCOMP_FLUXES[star_index[l]].min())
   
    '''
    for l in range(0,k):
        for r in range(0,size):
            copy_COMP_FLUXES[star_index[l]][r] = COMP_FLUXES[star_index[l]][r] / np.sum(COMP_FLUXES[star_index,r])
            
    for l in range(0,k):
        for r in range(0,Msize):
            copy_MCOMP_FLUXES[star_index[l]][r] = MCOMP_FLUXES[star_index[l]][r] / np.sum(MCOMP_FLUXES[star_index,r])
    '''
    
    '''
    # Normalize rel. flux of selected comp. stars with MAX
    for i in range(0,k):
        COMP_FLUXES[star_index[i]]  =  COMP_FLUXES[star_index[i]] / MAX
        MCOMP_FLUXES[star_index[i]] = MCOMP_FLUXES[star_index[i]] / MAX
    '''
    
    fig, axs = plt.subplots(k,2,sharex=False,gridspec_kw={'width_ratios':[4,1]})

    label1 =  r'$\mathrm{raw\ data}$'
    label2 =  r'$\mathrm{filtered\ data}$'
    #fig.subplots_adjust(hspace=0)
    i = 0
    j = star_index[i]
    axs[i,0].plot(BJD,copy_COMP_FLUXES[j],marker='o',color='red',markersize=dot_size*1.,linestyle='None',label=label1)
    axs[i,0].plot(MBJD,copy_MCOMP_FLUXES[j],marker='o',color='gray',markersize=dot_size,linestyle='None',label=label2)
    plt.sca(axs[i,0])
    axs[i,0].set_ylim([0.,1.])
    plt.xticks(np.arange(BJD.min(),BJD.max(),step = (BJD.max() - BJD.min()) / 5.),[])
    plt.yticks(np.arange(0., 1.25, step=0.25),[0,'',0.5,'',1.])
    axs[i,0].set_xlim([BJD.min(),BJD.max()])
    #plt.legend(loc='upper center',bbox_to_anchor=(0,1.))
    axs[i,0].grid(True)

    plt.sca(axs[i,1])
    axs[i,1].set_ylim([0.,1.])
    plt.yticks(np.arange(0., 1.25, step=0.25),[0,'',0.5,'',1.])
    axs[i,1].yaxis.tick_right()
    axs[i,1].set_xticks([])
    axs[i,1].bar(['rms','IQR','$\eta^{-1}$','$\chi^2$'],yindices[i],width=0.5,color=['lightgray','darkgray','dimgray','black'])
    axs[i,1].set_aspect(4,anchor='SW')

    for i in range(1,k):
        j = star_index[i]
        axs[i,0].plot(BJD,copy_COMP_FLUXES[j],marker='o',color='red',markersize=dot_size,linestyle='None')
        axs[i,0].plot(MBJD,copy_MCOMP_FLUXES[j],marker='o',color='gray',markersize=dot_size,linestyle='None')
        plt.sca(axs[i,0])
        axs[i,0].set_ylim([0.,1.])
        plt.yticks(np.arange(0., 1., step=0.25),[0,'',0.5,''])
        if i < k - 1: plt.xticks(np.arange(BJD.min(),BJD.max(),step = (BJD.max() - BJD.min()) / 5.),[])
        else: plt.xticks(np.arange(BJD.min(),BJD.max(),step = (BJD.max() - BJD.min()) / 5.)) 
        plt.yticks(np.arange(0., 1., step=0.25),[0,'',0.5,''])
        axs[i,0].set_xlim([BJD.min(),BJD.max()])
        axs[i,0].grid(True)
        
        if i == k - 1:
            plt.xlabel(r'$\mathrm{BJD}$')
        
        plt.sca(axs[i,1])
        axs[i,1].set_ylim([0.,1.])
        plt.yticks(np.arange(0., 1., step=0.25),[0,'',0.5,''])
        axs[i,1].yaxis.tick_right()
        axs[i,1].bar(['$\mathbf{\sigma}$','$\mathrm{IQR}$','$\eta^{-1}$','$\mathbf{\chi}^2$'],yindices[i],width=0.5,color=['lightgray','darkgray','dimgray','black'])
        axs[i,1].set_aspect(4,anchor='SW')
        
        if i != k - 1: 
            axs[i,1].set_xticks([])
        
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel(r'$\mathrm{(selected) \ comp. \ stars \ norm. \ rel. \ flux}$',labelpad=15.)

    fig.subplots_adjust(hspace=0.,wspace = 0.1)
    #plt.legend()
    fig = plt.gcf() # get current figure
    fig.set_size_inches(common_width * 0.75*s,1.4 * s * k)
    # Base names:
    base_ = 'comp_stars_and_var_indices'
    # Complete file names:
    plot_name = write.file_name(prefix_,base_,suffix_) + extension
    fig.savefig(plot_name, dpi=400, bbox_inches='tight')
    end = 'Plot (comp. stars with var. indices) saved as:\n' + plot_name + '\n'
    print(end)
    

plt.show()


