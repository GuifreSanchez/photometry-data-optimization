import math
import time
import numpy as np
import read
import matplotlib.pyplot as plt
import varindex as vi
import filter_data as fd
import more_stats as st
import bar
# Computation of new flux on target and comp. stars, given a certain
# configuration (config).

# INPUT
# config    : list of 1s and 0s of length #{comp. stars}. 1 -> discarded, 0 -> selected. 
# rel_target: np.array of rel. flux for target. Dimensions: 1 x #{data points}.
# rel_comp  : np.array of ALL rel. fluxes for comp. stars. Dimensions: #{comp. stars} x #{data points}.

# OUTPUT: a list with 3 elements.
# 1. new_rel  : np.array w/ new rel. flux for target star. 
# 2. new_rel_c: np.array w/ new rel. flux for all selected comp. stars (w.r.t. to chosen config.).
# 3. new_c    : list of original indices for selected comp. stars, in case they are needed a posteriori. 
def new_flux(rel_target,rel_comp,config):
    n = len(rel_target[0,:])
    c = len(rel_comp[:,0])
    new_rel_target = np.zeros((1,n))
    new_c = []
    for i in range(0,c):
        if config[i] == 0: new_c.append(i)
    new_rel_c = np.zeros((len(new_c),n))
    selected = len(new_c)
    # we think there is one target and several comparison stars
    for i in range(0,n):
        mod_rel_flux   = 0.
        mod_rel_flux_c = 0.
        for j in range(0,c):
            mod_rel_flux += config[j] * rel_comp[j][i] / (1. + rel_comp[j][i])
        for j in range(0,selected):
            new_rel_c[j][i] = rel_comp[new_c[j]][i] / (1. - (1. + rel_comp[new_c[j]][i])*mod_rel_flux)
        new_rel_target[0][i] = rel_target[0][i] / (1. - mod_rel_flux)
        
    result = [new_rel_target,new_rel_c,new_c]
    return result

# Updated errors for new comp. star config. 
# We use classical error propagation.

# INPUT 
# rel_target,rel_comp: see previous functions. 
# rel_comp_err       : np.array w/ errors from AstroImageJ corresponding to target star rel. fluxes. 
#                      Dimensions: 1 x #{data points}
# rel_comp_err       : np.array w/ errors from AstroImageJ corresponding to ALL comp. stars rel. fluxes.
#                      Dimensions: #{comp. stars} x #{data points}.
# config             : selected configuration. 

# OUTPUT: a list with 2 elements. 
# new_rel_t_err: np.array w/ all the updated errors corresponding to the target star with the new config. 
#                Dimensions: 1 x #{data points}
# new_rel_c_err: np.array w/ all the updated errors, corresponding to the selected comp. stars in config.
#                Dimensions: #{0s in config. / selected comp. stars} x #{data points}.
# [Obs.: new_rel_c_err[2][0] is the rel. flux error in the first .FITS of the 3rd selected comp. star 
#  according to the considered config list. If, for example, config = [0,1,0,1,0,0], new_rel_c_err[2][0]
#  will be referencing the updated error of comp. star 4, w.r.t. original indexation.]
def new_flux_err(rel_target,rel_target_err,rel_comp,rel_comp_err,config):
    n = len(rel_comp[0,:])
    c = len(rel_comp[:,0])
    new_c = []
    disc_c = []
    for i in range(0,c):
        if config[i] == 0: new_c.append(i)
        else: disc_c.append(i)
        
    discarded = len(disc_c)
    new_rel_c = np.zeros((len(new_c),n))
    new_rel_c_err = np.zeros((len(new_c),n))
    selected = len(new_c)
    new_rel_c = new_flux(rel_target,rel_comp,config)[1]
    # we think there is one target and several comparison stars

    for i in range(0,selected):
        for l in range(0,n):
            old_phi = rel_comp[new_c[i]][l]
            new_phi = new_rel_c[i][l]
            dphi1 = 0.
            for j in range(0,c):
                dphi1 += config[j] * rel_comp[j][l] / (1 + rel_comp[j][l])
            dphi1 *= new_phi**2 / old_phi
            dphi1 += new_phi    / old_phi
            dphi2 = np.zeros((1,discarded))
            for k in range(0,discarded):
                dphi2[0][k] = new_phi**2 / old_phi * ((1 + old_phi)/(1 + rel_comp[disc_c[k]][l]) - (1 + old_phi)*rel_comp[disc_c[k]][l] / (1 + rel_comp[disc_c[k]][l])**2)
            err12 = dphi1**2*rel_comp_err[new_c[i]][l]**2
            err22 = 0.
            for k in range(0,discarded):
                err22 += dphi2[0][k]**2*rel_comp_err[disc_c[k]][l]**2
            new_rel_c_err[i][l] = math.sqrt(err12 + err22)
        
    new_rel_t = new_flux(rel_target,rel_comp,config)[0]
    new_rel_t_err = np.zeros((1,n),dtype=np.float64)
    
    for i in range(0,n):
        err12 = (new_rel_t[0][i] / rel_target[0][i] * rel_target_err[0][i])**2
        err22 = 0.
        for k in range(0,discarded):
            err22 += ((new_rel_t[0][i] / (1 + rel_comp[disc_c[k]][i]))**2 / rel_target[0][i] * rel_comp_err[disc_c[k]][i])**2
        new_rel_t_err[0][i] = math.sqrt(err12 + err22)
    
    result = [new_rel_t_err,new_rel_c_err]
    return result

# Given a list with integers, returns a list with same entries
# which is the next permutation of such elements assuming lexicographic order. 
def next_permutation(a):
    """Generate the lexicographically next permutation inplace.

    https://en.wikipedia.org/wiki/Permutation#Generation_in_lexicographic_order
    Return false if there is no next permutation.
    """
    # Find the largest index i such that a[i] < a[i + 1]. If no such
    # index exists, the permutation is the last permutation
    for i in reversed(range(len(a) - 1)):
        if a[i] < a[i + 1]:
            break  # found
    else:  # no break: not found
        return False  # no next permutation

    # Find the largest index j greater than i such that a[i] < a[j]
    j = next(j for j in reversed(range(i + 1, len(a))) if a[i] < a[j])

    # Swap the value of a[i] with that of a[j]
    a[i], a[j] = a[j], a[i]

    # Reverse sequence from a[i + 1] up to and including the final element a[n]
    a[i + 1:] = reversed(a[i + 1:])
    return(a) 

# Computes rms for a concrete number of configurations (steps), and returns
# also the associated lists corresponding to each config. 

# INPUT
# rel_target: np. array w/ (original) rel. flux for the target.
# rel_comp  : np. array w/ ALL rel. fluxes for comp. stars. 
# config0   : list of 1s and 0s corresponding to the starting configuration (see comments in new_flux). 
# steps     : # of configurations we want to run through. For each one the rms of the target's rel. flux will
#             be computed.

# OUTPUT: a list with 2 elements.
# 1. result   : np.array with all the computed rmss. Dimensions: 1 x steps.
# 2. all_perms: np.array with all considered configs. Dimensions: steps x #{data points}.
def rms_search(rel_target,rel_target_err,rel_comp,rel_comp_err,config0,steps):
    perm = config0
    weighted = True
    result = np.zeros((1,steps),dtype=np.float64)
    all_perms = np.zeros((steps,len(rel_comp[:,0])))
    for i in range(0,steps):
        target_flux     = new_flux(rel_target,rel_comp,perm)[0]
        if weighted == True:
            target_flux_err = new_flux_err(rel_target,rel_target_err,rel_comp,rel_comp_err,perm)[0]
            rms_target_flux = vi.rms(target_flux[0],target_flux_err[0],weighted)
        else: 
            rms_target_flux = vi.rms(target_flux[0],rel_target_err[0],weighted)
        result[0][i] = rms_target_flux
        all_perms[i,:] = np.asarray(perm)
        perm = next_permutation(perm)
    return [result,all_perms]

# Computes rms for a concrete number of configurations (steps), fixing 
# as discarded certain comp. stars (fixed_config). Returns also the associated
# lists corresponding to each config. under consideration. 

# INPUT 
# rel_target, rel_comp, config0, steps: see previous function. 
# fixed_config                        : list with 1s and 0s. 1s -> stars that will remain discarded through
#                                       all configs considered.

# OUTPUT: a list with 2 elements (same as previous function).
def rms_search_mod(rel_target,rel_target_err,rel_comp,rel_comp_err,config0,fixed_config,steps):
    weighted = True
    c = len(rel_comp[:,0])
    # since discarded stars from fixed_config remain discarded
    # through all the computations, we just take the part of
    # config0 that is "allowed" to change. We consider the next
    # (steps) permutation from the modified/reduced config0. 
    for i in range(0,c): 
        if fixed_config[i] == 1: config0.pop(i - (c - len(config0)))
    perm = config0
    result = np.zeros((1,steps),dtype=np.float64)
    all_perms = np.zeros((steps,c))
    for i in range(0,steps):
        perm_ext = np.asarray(perm)
        for j in range(0,c):
            if fixed_config[j] == 1: perm_ext = np.insert(perm_ext,j,1) 
        target_flux     = new_flux(rel_target,rel_comp,perm_ext)[0]
        if weighted == True:
            target_flux_err = new_flux_err(rel_target,rel_target_err,rel_comp,rel_comp_err,perm_ext)[0]
            rms_target_flux = vi.rms(target_flux[0],target_flux_err[0],weighted)
        else: 
            rms_target_flux = vi.rms(target_flux[0],rel_target_err[0],weighted)
        result[0][i] = rms_target_flux
        all_perms[i,:] = np.asarray(perm_ext)
        perm = next_permutation(perm)
    return [result,all_perms]


# Returns combinatorial number n choose k. 
def comb(n,k):
    result = math.factorial(n) / (math.factorial(n - k) * math.factorial(k))
    return(result)
        

# Computes N_min minimum rmss from all possible configs. w/ k discarded comp. stars. 
# Returns also the configs that realize such rmss.

# INPUT
# rel_target: see previous functions. 
# rel_comp  : see previous functions. 
# k         : number of disc. comp. stars. 
# N_min     : numbers of best (minimum) rmss values we want to keep. 

# OUTPUT: a list with 2 elements.
# 1. all_min_rms: np.array w/ the computed values for the min. rmss. Dimensions: 1 x N_min.
# 2. all_configs: np.array w/ all configurations corresponding to the computed rmss. Dimensions: N_min x #{comp. stars}.
def best_comp_rms(rel_target,rel_target_err,rel_comp,rel_comp_err,k,N_min):
    c = len(rel_comp[:,0])
    start_config = [0 for _ in range(0,c)]
    for i in range(0,k): start_config[c - i - 1] = 1
    c_choose_k = int(comb(c,k))
    rmss = rms_search(rel_target,rel_target_err,rel_comp,rel_comp_err,start_config,c_choose_k)
    all_min_rms = np.zeros((1,N_min),dtype=np.float64)
    all_indices = np.zeros((1,N_min))
    all_configs = np.zeros((N_min,c))
    ind_min_rmss = np.argsort(rmss[0][0,:])
    
    for i in range(0,N_min):
        all_min_rms[0][i] = rmss[0][0][ind_min_rmss[i]]
        all_indices[0][i] = ind_min_rmss[i]
        all_configs[i,:]  = rmss[1][ind_min_rmss[i],:]
    
    result = [all_min_rms,all_configs]
    
    return result
        
    

# Essentially same function as best_comp_rms but using rms_search_mod
# instead of rms_search, i.e. a number of comp. stars remain discarded through
# all configurations (fixed_config).
def best_comp_rms_mod(rel_target,rel_target_err,rel_comp,rel_comp_err,fixed_config,k,N_min):
    local_N_min = N_min
    c = len(rel_comp[:,0])
    m = int(np.sum(np.asarray(fixed_config)))
    start_config = [0 for _ in range(0,c - m)]
    for i in range(0,k - m): start_config[c - m - i - 1] = 1
    for i in range(0,c): 
        if fixed_config[i] == 1: start_config.insert(i,1)

    max_configs = int(comb(c - m,k - m))
    if local_N_min > max_configs: local_N_min = max_configs
    rmss = rms_search_mod(rel_target,rel_target_err,rel_comp,rel_comp_err,start_config,fixed_config,max_configs)
    all_min_rms = np.zeros((1,local_N_min),dtype=np.float64)
    all_indices = np.zeros((1,local_N_min))
    all_configs = np.zeros((local_N_min,c))
    
    ind_min_rmss = np.argsort(rmss[0][0,:])
    
    for i in range(0,local_N_min):
        all_min_rms[0][i] = rmss[0][0][ind_min_rmss[i]]
        all_indices[0][i] = ind_min_rmss[i]
        all_configs[i,:]  = rmss[1][ind_min_rmss[i],:]
        
    result = [all_min_rms,all_configs]
    
    return result


# Given a set of configurations (as 0s and 1s - lists) returns a list where at position
# i a 1 indicates that comp. star i was discarded in all lists in the original set, and 
# a 0 indicates the contrary (in some list in configs comp. star i was NOT discarded).
def keep(configs):
    n = len(configs[:,0])
    result = configs[0,:]
    for i in range(1,n):
        result = np.multiply(result,configs[i,:])
        
    return result

# Computes the logical sum of an np.array containing
# several comp. star configurations.
# e.g.: if input is [[0,0,0,1],[0,1,0,1]] add() returns [0,1,0,1].
def add(configs):
    n = len(configs[:,0])
    m = len(configs[0,:])
    result = np.zeros((1,m))[0]
    for i in range(0,m):
        result[i] = np.sum(configs[:,i])
        if result[i] != 0: result[i] = 1
        
    return result


# Computes best target rmss for a set of comp. stars in 2 different ways:
# I.  Using best_comp_rms_mod we compute the best rms for the target star when
#     only 1 comp. is discarded. 
#     Using the config. obtained in the previous step as fixed_config, we compute
#     the best 2 rmss w/ 2 disc. comp. stars. 
#     Using the keep function, we create a fixed_config with the 2 best configs obtained
#     in the previous step. 
#     Iterate this process for best i rmss w/ i disc. comp stars for i <= k_max.
# II. Same idea as I. but in this case the number of min. rmss computed is fixed: N_min.

# INPUT
# rel_target,rel_comp,N_min: see previous functions. 
# const                    : False -> method I. True -> method II. 
# k_max                    : see description of I. 
# init_config              : initial comp. star configuration.

# OUTPUT: a list with k_max elements. 
# element i: same form as output for best_comp_rms_mod (best rmss and configs at iteration step i).
def min_rms(rel_target,rel_target_err,rel_comp,rel_comp_err,const,init_config,k_max,N_min):
    target_data     = rel_target
    target_data_err = rel_target_err
    comp_data       = rel_comp
    comp_data_err   = rel_comp_err
    c = len(rel_comp[:,0])
    k_max = int(k_max)
    m = int(np.sum(np.asarray(init_config)))
    results = [[] for _ in range(0,k_max - m)]
    if const == False:
        m = int(np.sum(np.asarray(init_config)))
        fixed_cfg = init_config
        results[0] = best_comp_rms_mod(target_data,target_data_err,comp_data,comp_data_err,fixed_cfg,m + 1,m + 1)
        bar.printProgressBar(0, k_max - m, prefix_ = 'Progress:', suffix_ = 'Complete', length = 50)
        for i in range(m + 1,k_max):
            bar.printProgressBar(i - m + 1, k_max - m, prefix_ = 'Progress:', suffix_ = 'Complete', length = 50)
            fixed_cfg = keep(results[i - 1 - m][1])
            results[i - m] = best_comp_rms_mod(target_data,target_data_err,comp_data,comp_data_err,fixed_cfg,i + 1,i + 1 - m)
    else:
        m = int(np.sum(np.asarray(init_config)))
        fixed_cfg = init_config
        results[0] = best_comp_rms_mod(target_data,target_data_err,comp_data,comp_data_err,fixed_cfg,m + 1,N_min)
        for i in range(m + 1,k_max):
            bar.printProgressBar(i - m + 1, k_max - m, prefix_ = 'Progress:', suffix_ = 'Complete', length = 50)
            fixed_cfg = keep(results[i - 1][1])
            results[i - m] = best_comp_rms_mod(target_data,target_data_err,comp_data,comp_data_err,fixed_cfg,i + 1,N_min)
        
    return results


# Computes configuration according to IQR values for selected comparison stars
# in two different ways (depending on the value of opt):
# if opt == True : output contains a configuration where the worst k comp. stars
#                  w.r.t. IQR index (i.e. those with highest IQR) are discarded. 
# if opt == False: output contains a configuration where comp. stars with IQR > threshold
#                  have been discarded. 

# INPUT
# rel_flux : np.array corresponding to the rel. fluxes of the comp. stars 
#            (see REL_FLUX_C, photometry, read.py module).
# opt      : bool-type variable, see description above.
# k        : integer >= 0, see description above. 
# threshold: threshold for selection method corresponding to opt == False. 

# OUTPUT: list with 2 elements.
# 1. config : list with 0s (selected comp. stars) and 1s (discarded comp. stars)
# 2. IQRs[0]: np.array with all IQR index values for the different comp. stars. 
def select_IQR(rel_flux,opt,k,threshold):
    c = len(rel_flux[:,0])
    IQRs = np.zeros((1,c),dtype=np.float64)
    config = [0 for _ in range(0,c)]
    for i in range(0,c):
        IQRs[0][i] = vi.IQR(rel_flux[i],threshold)[0]
    if opt == True:
        iqr_values = IQRs[0]
        ind_max_IQR = np.argsort(iqr_values)
        config = [0 for _ in range(0,c)]
        for i in range(0,k):
            config[ind_max_IQR[c - 1 - i]] = 1
    else:
        for i in range(0,c):
            if vi.IQR(rel_flux[i],threshold)[2] == True: config[i] = 1
            
    return [config,IQRs[0]]

# The functions select_rms, select_chi2, select_neumann are analogous to select_IQR
# but using variability indices: rms, chi2 and neumann resp., corresponding to the
# functions with same name in the varindex.py module.
def select_rms(rel_flux,rel_flux_err,opt,k,threshold):
    weighted = True
    c = len(rel_flux[:,0])
    rmss = np.zeros((1,c),dtype=np.float64)
    config = [0 for _ in range(0,c)]
    for i in range(0,c):
        rmss[0][i] = vi.rms(rel_flux[i],rel_flux_err[i],weighted)
    if opt == True:
        rms_values = rmss[0]
        ind_max_rms = np.argsort(rms_values)
        config = [0 for _ in range(0,c)]
        for i in range(0,k):
            config[ind_max_rms[c - 1 - i]] = 1
    else:
        for i in range(0,c):
            if vi.rms(rel_flux[i],rel_flux_err[i],weighted) > threshold: config[i] = 1
            
    return [config,rmss[0]]

def select_neumann(rel_flux,opt,k,threshold):
    c = len(rel_flux[:,0])
    etas = np.zeros((1,c),dtype=np.float64)
    config = [0 for _ in range(0,c)]
    for i in range(0,c):
        etas[0][i] = vi.neumann(rel_flux[i],threshold)[0]
    if opt == True:
        eta_values = etas[0]
        ind_max_eta = np.argsort(eta_values)
        config = [0 for _ in range(0,c)]
        for i in range(0,k):
            config[ind_max_eta[c - 1 - i]] = 1
    else:
        for i in range(0,c):
            if vi.neumann(rel_flux[i],threshold)[1] == True: config[i] = 1
            
    return [config,etas[0]]

def select_chi2(rel_flux,rel_flux_err,opt,k,alpha):
    c = len(rel_flux[:,0])
    chi2s = np.zeros((1,c),dtype=np.float64)
    config = [0 for _ in range(0,c)]
    for i in range(0,c):
        chi2s[0][i] = vi.chi2(rel_flux[i],rel_flux_err[i],alpha)[0]
    if opt == True:
        chi2_values = chi2s[0]
        ind_max_chi2 = np.argsort(chi2_values)
        config = [0 for _ in range(0,c)]
        for i in range(0,k):
            config[ind_max_chi2[c - 1 - i]] = 1
    else:
        for i in range(0,c):
            if vi.chi2(rel_flux[i],rel_flux_err[i],alpha)[2] == True: config[i] = 1
            
    return [config,chi2s[0]]

# Computes combined config. from configs. corresponding to selected variability indices
# obtained using select_ type functions (see previous 4 functions).

# INPUT
# rel_flux, rel_flux_err: as in the select_ type functions. 
# opts                  : list with 4 bool-type elements corresponding to opt parameter
#                         in select_rms / IQR / neumann / chi2 (in this order). 
# ks                    : list with 4 integers >= 0 corresponding to the k parameter
#                         in select_rms / IQR / neumann / chi2 (in this order). 
# thresholds            : list with 4 floats corresponding to the threshold parameter
#                         in select_rms / IQR / neumann and the alpha parameter in select_chi2.
# arr                   : list with 4 bool-type elements.
#                         arr[i] = True  -> index i is selected. 
#                         arr[i] = False -> index i is not selected.
#                         order - 0: rms, 1: IQR, 2: neumann, 3: chi2.

# OUTPUT: config. list that combines the configs. obtained using select_(?) for the selected
#         indices according to arr. Parameters for these select_ functions are taken from
#         input variables opts, ks, thresholds. 
def select_combined(rel_flux,rel_flux_err,opts,ks,thresholds,arr): 
    c = len(rel_flux[:,0])
    count = 0 
    for i in range(0,4):
        if arr[i] == False: count += 1
        
    if count == 4: 
        return np.zeros((1,c))[0]
    else:
    
        rms_config     = select_rms(rel_flux,rel_flux_err,opts[0],ks[0],thresholds[0])[0]
        iqr_config     = select_IQR(rel_flux,opts[1],ks[1],thresholds[1])[0]
        neumann_config = select_neumann(rel_flux,opts[2],ks[2],thresholds[2])[0]
        chi2_config    = select_chi2(rel_flux,rel_flux_err,opts[3],ks[3],thresholds[3])[0]
        
        all_configs = [rms_config,iqr_config,neumann_config,chi2_config]
        sel_configs = []
        for i in range(0,4): 
            if arr[i] == True: sel_configs.append(all_configs[i])
            
        n = len(sel_configs)
        np_sel_configs = np.zeros((n,c))
        for i in range(0,n):
            np_sel_configs[i] = sel_configs[i]
        
        return keep(np_sel_configs)

# STAR SELECTION BASED ON (OTHER) SIGNAL-TO-NOISE RATIO MEASURES
def snr_search_mod(rel_target,rel_target_err,rel_comp,rel_comp_err,config0,fixed_config,steps):
    weighted = True
    c = len(rel_comp[:,0])
    # since discarded stars from fixed_config remain discarded
    # through all the computations, we just take the part of
    # config0 that is "allowed" to change. We consider the next
    # (steps) permutation from the modified/reduced config0. 
    for i in range(0,c): 
        if fixed_config[i] == 1: config0.pop(i - (c - len(config0)))
    perm = config0
    result = np.zeros((1,steps),dtype=np.float64)
    all_perms = np.zeros((steps,c))
    for i in range(0,steps):
        perm_ext = np.asarray(perm)
        for j in range(0,c):
            if fixed_config[j] == 1: perm_ext = np.insert(perm_ext,j,1) 
        target_flux     = new_flux(rel_target,rel_comp,perm_ext)[0]
        if weighted == True:
            target_flux_err = new_flux_err(rel_target,rel_target_err,rel_comp,rel_comp_err,perm_ext)[0]
            snr_target_flux = vi.snr(target_flux[0],target_flux_err[0],weighted)
        else: 
            snr_target_flux = vi.snr(target_flux[0],rel_target_err[0],weighted)
        result[0][i] = snr_target_flux
        all_perms[i,:] = np.asarray(perm_ext)
        perm = next_permutation(perm)
    return [result,all_perms]

def best_comp_snr_mod(rel_target,rel_target_err,rel_comp,rel_comp_err,fixed_config,k,N_min):
    local_N_min = N_min
    c = len(rel_comp[:,0])
    m = int(np.sum(np.asarray(fixed_config)))
    start_config = [0 for _ in range(0,c - m)]
    for i in range(0,k - m): start_config[c - m - i - 1] = 1
    for i in range(0,c): 
        if fixed_config[i] == 1: start_config.insert(i,1)

    max_configs = int(comb(c - m,k - m))
    if local_N_min > max_configs: local_N_min = max_configs
    snrs = snr_search_mod(rel_target,rel_target_err,rel_comp,rel_comp_err,start_config,fixed_config,max_configs)
    
    all_min_snrs = np.zeros((1,local_N_min),dtype=np.float64)
    all_indices = np.zeros((1,local_N_min))
    all_configs = np.zeros((local_N_min,c))
    
    ind_min_snrs = np.argsort(snrs[0][0,:])
    
    for i in range(0,local_N_min):
        all_min_snrs[0][i] = snrs[0][0][ind_min_snrs[i]]
        all_indices[0][i] = ind_min_snrs[i]
        all_configs[i,:]  = snrs[1][ind_min_snrs[i],:]
        
    result = [all_min_snrs,all_configs]
    
    return result

def min_snr(rel_target,rel_target_err,rel_comp,rel_comp_err,const,init_config,k_max,N_min):
    target_data     = rel_target
    target_data_err = rel_target_err
    comp_data       = rel_comp
    comp_data_err   = rel_comp_err
    c = len(rel_comp[:,0])
    m = int(np.sum(np.asarray(init_config)))
    results = [[] for _ in range(0,k_max - m)]
    if const == False:
        m = int(np.sum(np.asarray(init_config)))
        fixed_cfg = init_config
        results[0] = best_comp_snr_mod(target_data,target_data_err,comp_data,comp_data_err,fixed_cfg,m + 1,1)
        bar.printProgressBar(0, k_max - m, prefix_ = 'Progress:', suffix_ = 'Complete', length = 50)
        for i in range(m + 1,k_max):
            bar.printProgressBar(i - m + 1, k_max - m, prefix_ = 'Progress:', suffix_ = 'Complete', length = 50)
            fixed_cfg = keep(results[i - 1 - m][1])
            results[i - m] = best_comp_snr_mod(target_data,target_data_err,comp_data,comp_data_err,fixed_cfg,i + 1,i + 1 - m)
    else:
        m = int(np.sum(np.asarray(init_config)))
        fixed_cfg = init_config
        results[0] = best_comp_snr_mod(target_data,target_data_err,comp_data,comp_data_err,fixed_cfg,m + 1,N_min)
        for i in range(m + 1,k_max):
            bar.printProgressBar(i - m + 1, k_max - m, prefix_ = 'Progress:', suffix_ = 'Complete', length = 50)
            fixed_cfg = keep(results[i - 1][1])
            results[i - m] = best_comp_snr_mod(target_data,target_data_err,comp_data,comp_data_err,fixed_cfg,i + 1,N_min)
        
    return results
