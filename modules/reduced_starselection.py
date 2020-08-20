import math
import numpy as np
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
# First results seem to be of order of old_errors. Some revision may be needed. 
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
