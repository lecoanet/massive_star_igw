
import numpy as np
import matplotlib

data = np.loadtxt('L_GYRE.dat')

freq = data[0,:]
transfer_ell1 = data[1,:]
transfer_ell2 = data[2,:]
transfer_ell3 = data[3,:]

b_ell = np.array([0.708,0.325,0.0626])
ell = np.array([1.,2.,3.])
Lambda = np.sqrt(ell*(ell+1))

spectrum_R = np.sum( b_ell[:,None]*data[1:]*Lambda[:,None]**(-0.9),axis=0) * freq**(0.4)
spectrum_LQ = np.sum( b_ell[:,None]*data[1:]*Lambda[:,None]**( 5/2),axis=0) * freq**(-3.25)

data_234517653 = np.loadtxt('EPIC234517653.dat')
freq_obs = data_234517653[0]

freq_obs = freq_obs[(freq_obs > 0.05) & (freq_obs < 1.85)]
df = freq_obs[1] - freq_obs[0]

spectrum_R_filter = np.zeros(len(freq_obs))
spectrum_LQ_filter = np.zeros(len(freq_obs))

mask = (freq > freq_obs[0] - df/2) & (freq < freq_obs[0] + df/2)
spectrum_R_filter[0] = np.max(spectrum_R[mask])
spectrum_LQ_filter[0] = np.max(spectrum_LQ[mask])
for i in range(1,len(freq_obs)-1):
  mask = (freq > (freq_obs[i] + freq_obs[i-1])/2) & (freq < (freq_obs[i] + freq_obs[i+1])/2)
  spectrum_R_filter[i] = np.max(spectrum_R[mask])
  spectrum_LQ_filter[i] = np.max(spectrum_LQ[mask])

mask = (freq > freq_obs[-1] - df/2) & (freq < freq_obs[-1] + df/2)
spectrum_R_filter[-1] = np.max(spectrum_R[mask])
spectrum_LQ_filter[-1] = np.max(spectrum_LQ[mask])

data = np.vstack([freq_obs,spectrum_R_filter,spectrum_LQ_filter])
np.savetxt('spectra.dat',data)


