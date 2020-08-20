import math
import numpy as np
import read
import filter_data as fd
import more_stats as st
import scipy.stats as pyst
import matplotlib.pyplot as plt

# Given an np.array (arr) computes its (normalized) root mean square error w.r.t. the mean (i.e. RSD / weighted RSD). 
def rms(flux,err,weighted):
    if weighted == False:
        m = np.mean(flux)
        result = np.sqrt(np.mean((flux - m)**2))
        result /= m
        return result
    else:
        wmean   = np.sum(flux*err**(-2)) / np.sum(err**(-2))
        wstd    = np.sum(err**(-2)*(flux - wmean)**2)
        wfactor = np.sum(err**(-2)) / (np.sum(err**(-2))**2 - np.sum(err**(-4)))
        wstd    = math.sqrt(wfactor*wstd)
        result  = wstd / wmean
        return result

# Chi^2 test for comp. star rel. flux where H0 = star has constant flux. 

# INPUT 
# flux : np.array of dimensions 1 x #{data points} corresponding to rel. flux one of the comp. stars. 
# err  : np.array of same dimensions as flux, corresponding to rel. flux error of the previous array. 
# alpha: 1 - alpha = statistical confidence for the test. 

# OUTPUT: a list with 3 elements
# stat          : empirical value of the chi^2 statistic. 
# critical_value: chi^2 dist. critical value corresponding to the confidence level (alpha) considered.
# variable      : True  -> H0 is rejected (the star can be classified as variable). 
#                 False -> H0 is NOT rejected (there is no enough evidence to conclude 
#                          that the star is variable). 
# [Obs.: for low alpha, variable = False may allow us to conclude that the star is NOT variable].
def chi2(flux,err,alpha):
    N = len(flux)
    stat = 0.
    wmean = np.sum(flux*err**(-2)) / np.sum(err**(-2))
    stat  = np.sum((flux - wmean)**2 * err**(-2))
    critical_value = pyst.chi2.ppf(1.-alpha,N - 1)
    variable = False
    if critical_value < stat: variable = True
    result = [stat/float(N) - 1.,critical_value,variable]
    return result

# Computation of inverse of von Neumann index (1 / eta) for variability detection

# INPUT
# flux: np.array w/ comp. star rel. flux. 
# c   : threshold. If 1 / eta > c, the star is classified as variable. 

# OUTPUT: a list with 2 elements. 
# stat    : computed value for inverse of von Neumann index, 1 / eta. 
# variable: False -> selected comp. star is declared non-variable. 
#           True  -> selected comp. star is declared variable. 
# [Obs.: first values of order ~ 1 as expected(?)]    
def neumann(flux,c):
    N = len(flux)
    delta2 = 0.
    m = np.mean(flux)
    for i in range(0,N - 1):
        delta2 += (flux[i + 1] - flux[i])**2
    delta2 /= (N - 1)
    sigma2 = np.var(flux,ddof=1,dtype=np.float64)
    
    eta = delta2/sigma2
    stat = 1. / eta
    variable = False
    if stat > c: variable = True
    return [stat,variable]

# Computation of the IQR (InterQuartile Range) index for variability detection 

# INPUT
# flux  : np.array w/ comp. star rel. flux. 
# norm_c: normalized (w.r.t. to Gaussian distr.) critical value. If IQR > c, the star is classified as variable. 

# OUTPUT: list with 3 elements. 
# norm_iqr: normalized IQR associated to flux (dividing by standard dev. of flux itself).
# raw_iqr : IQR of data in flux. 
# variable: False -> selected comp. star is declared non-variable. 
#           True  -> selected comp. star is declared variable. 
def IQR(flux,norm_c):
    raw_iqr = pyst.iqr(flux)
    norm_iqr = pyst.iqr(flux,scale='normal') / np.std(flux,ddof = 1,dtype=np.float64)
    variable = False
    if norm_iqr > norm_c: variable = True
    result = [norm_iqr,raw_iqr,variable]
    return result

# Index used in the min_snr function in the starselection.py module
def snr(flux,err,weighted):
    N = len(flux)
    if weighted == True:
        l1 = 0.
        wmean = np.sum(flux*err**(-2)) / np.sum(err**(-2))
        for i in range(0,N - 1):
            l1 += (flux[i] - wmean) * (flux[i + 1] - wmean)
        l1 /= np.sum((flux - wmean)**2)
        return l1
    else:
        l1 = 0.
        cmean = np.mean(flux)
        for i in range(0,N - 1):
            l1 += (flux[i] - cmean) * (flux[i + 1] - cmean)
        l1 /= np.sum((flux - cmean)**2)
        return l1


