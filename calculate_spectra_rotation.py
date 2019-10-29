
import numpy as np
import matplotlib
import csv

data = np.loadtxt('L_GYRE.dat')

freq = data[0,:]
transfer_ell1 = data[1,:]
transfer_ell2 = data[2,:]
transfer_ell3 = data[3,:]

b_ell = np.array([0.708,0.325,0.0626])
ell = np.array([1.,2.,3.])
Lambda = np.sqrt(ell*(ell+1))

spectrum_R1 = b_ell[0]*data[1]*Lambda[0]**(-0.9)*freq**(0.4)
spectrum_R2 = b_ell[1]*data[2]*Lambda[1]**(-0.9)*freq**(0.4)
spectrum_R3 = b_ell[2]*data[3]*Lambda[2]**(-0.9)*freq**(0.4)

spectrum_LQ1 = b_ell[0]*data[1]*Lambda[0]**(5/2)*freq**(-3.25)
spectrum_LQ2 = b_ell[1]*data[2]*Lambda[1]**(5/2)*freq**(-3.25)
spectrum_LQ3 = b_ell[2]*data[3]*Lambda[2]**(5/2)*freq**(-3.25)

data_234517653 = np.loadtxt('EPIC234517653.dat')
freq_obs = data_234517653[0]

freq_obs = freq_obs[(freq_obs > 0.05) & (freq_obs < 1.85)]
df = freq_obs[1] - freq_obs[0]

spectrum_R1_filter = np.zeros(len(freq_obs)-1)
spectrum_R2_filter = np.zeros(len(freq_obs)-1)
spectrum_R3_filter = np.zeros(len(freq_obs)-1)

spectrum_LQ1_filter = np.zeros(len(freq_obs)-1)
spectrum_LQ2_filter = np.zeros(len(freq_obs)-1)
spectrum_LQ3_filter = np.zeros(len(freq_obs)-1)

Om = 0.5

for i in range(1,len(freq_obs)-1):
    mask = (freq_obs[i] < 2*Om) & (freq > (freq_obs[i] + freq_obs[i-1])/2) & (freq < (freq_obs[i] + freq_obs[i+1])/2)
    if np.sum(mask) > 0:
      spectrum_R1_filter[i-1] = np.max(spectrum_R1[mask])
      spectrum_LQ1_filter[i-1] = np.max(spectrum_LQ1[mask])
    for j in [-1,0,1]:
      mask = (freq_obs[i] >= 2*Om) & (freq > (freq_obs[i] - j*Om + freq_obs[i-1])/2) & (freq < (freq_obs[i] - j*Om + freq_obs[i+1])/2)
      if np.sum(mask)>0:
        spectrum_R1_filter[i-1]+= 1/3*np.max(spectrum_R1[mask])
        spectrum_LQ1_filter[i-1]+= 1/3*np.max(spectrum_LQ1[mask])

for i in range(1,len(freq_obs)-1):
    mask = (freq_obs[i] < 2*Om) & (freq > (freq_obs[i] + freq_obs[i-1])/2) & (freq < (freq_obs[i] + freq_obs[i+1])/2)
    if np.sum(mask) > 0:
      spectrum_R2_filter[i-1] = np.max(spectrum_R2[mask])
      spectrum_LQ2_filter[i-1] = np.max(spectrum_LQ2[mask])
    for j in [-2,-1,0,1,2]:
      mask = (freq_obs[i] >= 2*Om) & (freq > (freq_obs[i] - j*Om + freq_obs[i-1])/2) & (freq < (freq_obs[i] - j*Om + freq_obs[i+1])/2)
      if np.sum(mask)>0:
        spectrum_R2_filter[i-1]+= 1/5*np.max(spectrum_R2[mask])
        spectrum_LQ2_filter[i-1]+= 1/5*np.max(spectrum_LQ2[mask])

for i in range(1,len(freq_obs)-1):
    mask = (freq_obs[i] < 2*Om) & (freq > (freq_obs[i] + freq_obs[i-1])/2) & (freq < (freq_obs[i] + freq_obs[i+1])/2)
    if np.sum(mask) > 0:
      spectrum_R3_filter[i-1] = np.max(spectrum_R3[mask])
      spectrum_LQ3_filter[i-1] = np.max(spectrum_LQ3[mask])
    for j in [-3,-2,-1,0,1,2,3]:
      mask = (freq_obs[i] >= 2*Om) & (freq > (freq_obs[i] - j*Om + freq_obs[i-1])/2) & (freq < (freq_obs[i] - j*Om + freq_obs[i+1])/2)
      if np.sum(mask)>0:
        spectrum_R3_filter[i-1]+= 1/7*np.max(spectrum_R3[mask])
        spectrum_LQ3_filter[i-1]+= 1/7*np.max(spectrum_LQ3[mask])

spectrum_R_filter = spectrum_R1_filter + spectrum_R2_filter + spectrum_R3_filter
spectrum_LQ_filter = spectrum_LQ1_filter + spectrum_LQ2_filter + spectrum_LQ3_filter

data = np.vstack([freq_obs[1:-1],spectrum_R_filter[:-1],spectrum_LQ_filter[:-1]])
np.savetxt('spectra_new_Om0p5.dat',data)

