import xlrd
import numpy as np

# Allows extraction of photometry data from a .xlsx file. 
# INPUT
# loc: .xlsx route in your PC. E.g.: loc = ("/home/guifress/pdo/data/TOI-1266_TJO_R.xlsx"). 
# org: list with 4 integers. org[0] = #{data points}, org[1] = #{target stars}, org[2] = #{comp. stars}, 
#                            org[3] = total number of stars. 
# ind: list with 2 elements. Each element is a list of integers corresponding to column numbers for...
#           i.   target stars rel. fluxes. 
#           ii.  comp. stars rel. fluxes. 
# [Obs.: in ind, we consider the first column of the .xlsx file to be column 1, NOT 0].

# OUTPUT: a list with 7 elements (np.arrays). #{data points} = dp
# 1. BJD.               Dim.: 1 x dp
# 2. AIRMASS.           Dim.: 1 x dp.
# 3. REL_FLUX_T.        Dim.: #{targets} x dp.
# 4. REL_FLUX_T_ERR.    Dim.: #{targets} x dp. 
# 5. REL_FLUX_SNR_T.    Dim.: #{targets} x dp.
# 6. REL_FLUX_C.        Dim.: #{comp. stars} x dp. 
# 7. REL_FLUX_C_ERR.    Dim.: #{comp. stars} x dp. 
# 8. REL_FLUX_SNR_C.    Dim.: #{comp. stars} x dp. 
def photometry(loc,org,ind):
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)

    size = org[0]
    targets = org[1]
    comparison = org[2]
    total = org[3]
    BJD = np.zeros((1,size))
    AIRMASS = np.zeros((1,size))
    REL_FLUX_T = np.zeros((targets,size))
    REL_FLUX_T_ERR = np.zeros((targets,size))
    
    REL_FLUX_C = np.zeros((comparison,size))
    REL_FLUX_C_ERR = np.zeros((comparison,size))
        
    REL_FLUX_SNR_T = np.zeros((targets,size))
    REL_FLUX_SNR_C = np.zeros((comparison,size))
    
    target_index = ind[0]
    target_err_index = []
    target_snr_index = []
    for i in range(0,targets): target_err_index.append(ind[0][i] +     total)
    for i in range(0,targets): target_snr_index.append(ind[0][i] + 2 * total)
    
    comp_index   = ind[1]
    comp_err_index = []
    comp_snr_index = []
    for i in range(0,comparison): comp_err_index.append(ind[1][i] +     total)
    for i in range(0,comparison): comp_snr_index.append(ind[1][i] + 2 * total)
    
    #print(comp_snr_index)

    for i in range(0,size):
        BJD[0][i] = sheet.cell_value(i + 1,8)
        AIRMASS[0][i] = sheet.cell_value(i + 1,9)
        for k in range(0,targets):
            REL_FLUX_T[k][i] = sheet.cell_value(i + 1,target_index[k] - 1)
            REL_FLUX_T_ERR[k][i] = sheet.cell_value(i + 1,target_err_index[k] - 1)
            REL_FLUX_SNR_T[k][i] = sheet.cell_value(i + 1,target_snr_index[k] - 1)
        for k in range(0,comparison):
            REL_FLUX_C[k][i] = sheet.cell_value(i + 1,comp_index[k] - 1)
            REL_FLUX_C_ERR[k][i] = sheet.cell_value(i + 1,comp_err_index[k] - 1)
            REL_FLUX_SNR_C[k][i] = sheet.cell_value(i + 1,comp_snr_index[k] - 1)

    result = [BJD,AIRMASS,REL_FLUX_T,REL_FLUX_T_ERR,REL_FLUX_SNR_T,REL_FLUX_C,REL_FLUX_C_ERR,REL_FLUX_SNR_C]
    return result


# Allows reading aperture peak pixel counts from both target and comparison stars. 
# Input variables are the same as in read.photometry.
def peak_values(loc,org,ind):
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)

    size = org[0]
    targets = org[1]
    comparison = org[2]
    total = org[3]
    
    T_PEAKS = np.zeros((targets,size))
    C_PEAKS = np.zeros((comparison,size))

    for i in range(0,size):
        for k in range(0,targets):
            star_number = ind[0][k] - ind[0][0] + 1
            if k == 0: T_PEAKS[k][i] = sheet.cell_value(i + 1,(ind[0][0] + total - 1) - 1 + 2 * total + 2 + 11)
            if k != 0: T_PEAKS[k][i] = sheet.cell_value(i + 1,(ind[0][0] + total - 1) - 1 + 2 * total + 2 + 20 + 10 + 18 * (star_number - 2))
            
        for k in range(0,comparison):
            star_number = ind[1][k] - ind[0][0] + 1
            C_PEAKS[k][i] = sheet.cell_value(i + 1,(ind[0][0] + total - 1) - 1 + 2 * total + 2 + 20 + 10 + 18 * (star_number - 2))
        
    return [T_PEAKS,C_PEAKS]
            

        
    
    
    
