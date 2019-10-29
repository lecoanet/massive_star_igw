
import numpy as np
import tomso as tomso
from tomso import gyre
import glob
import time
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg

# load modes

def read_modes(file_bases):
  freq_list = []
  n_g_list = []
  ell_list = []
  file_list = [file for base in file_bases for file in glob.glob('%s*.txt' %base)]

  for i,filename in enumerate(file_list):
    if i % 10 == 0: print(filename)
    header, data_mode = tomso.gyre.load_summary(filename)
    freq_list.append(-1j*(header['Refreq'] + 1j*header['Imfreq']))
    n_g_list.append(header['n_g'])
    ell_list.append(header['l'])
  
  freq = np.array(freq_list)
  n_g = np.array(n_g_list)
  ell = np.array(ell_list)

  return freq, n_g, ell

freq_ell1, n_g_ell1, ell_ell1 = read_modes(['../gyre_igw/modes_10/10XC066_ell1.'])
freq_ell2, n_g_ell2, ell_ell2 = read_modes(['../gyre_igw/modes_10/10XC066_ell2.'])
freq_ell3, n_g_ell3, ell_ell3 = read_modes(['../gyre_igw/modes_10/10XC066_ell3.'])

freq_ell1 = freq_ell1[-50:]
freq_ell2 = freq_ell2[-50:]
freq_ell3 = freq_ell3[-50:]

n_g_ell1 = n_g_ell1[-50:]
n_g_ell2 = n_g_ell2[-50:]
n_g_ell3 = n_g_ell3[-50:]

ell_ell1 = ell_ell1[-50:]
ell_ell2 = ell_ell2[-50:]
ell_ell3 = ell_ell3[-50:]

freq, n_g, ell = read_modes(['../gyre_igw/modes_10_ell_large/'])

f_ell1 = -freq_ell1.imag*1e-6*24*60*60
g_ell1 = -freq_ell1.real*1e-6*24*60*60
f_ell2 = -freq_ell2.imag*1e-6*24*60*60
g_ell2 = -freq_ell2.real*1e-6*24*60*60
f_ell3 = -freq_ell3.imag*1e-6*24*60*60
g_ell3 = -freq_ell3.real*1e-6*24*60*60

f = -freq.imag*1e-6*24*60*60
g = -freq.real*1e-6*24*60*60

f_all = np.concatenate([f_ell1,f_ell2,f_ell3,f])
g_all = np.concatenate([g_ell1,g_ell2,g_ell3,g])*2*np.pi
n_g_all = np.concatenate([n_g_ell1,n_g_ell2,n_g_ell3,n_g])
ell_all = np.concatenate([ell_ell1,ell_ell2,ell_ell3,ell])

mask = g_all > 0
f_all = f_all[mask]
g_all = g_all[mask]
n_g_all = n_g_all[mask]
ell_all = ell_all[mask]

g_mean = np.sqrt(g_all[:,None]*g_all[None,:])
f_sum = f_all[:,None] + f_all[None,:]
n_g_diff = np.abs(n_g_all[:,None] - n_g_all[None,:])

n_modes = len(f_all)

def gamma_min(f_target,n_g_target,ell_target):

    gamma_m = 1
    for i in range(n_modes):
      for j in range(i,n_modes):
        ell_j = ell_all[j]
        ell_i = ell_all[i]
        n_g_j = n_g_all[j]
        n_g_i = n_g_all[i]
        n_g_gtr = max(n_g_j,n_g_i)
        n_g_les = min(n_g_j,n_g_i)
        if (ell_j - ell_i <= ell_target) and (ell_j + ell_i >= ell_target):
            if ( np.abs(n_g_gtr - n_g_les - n_g_target) <= 2 ) or ( np.abs(n_g_gtr + n_g_les - n_g_target) <= 2 ):
                f_diff = np.abs(f_sum[i,j] - f_target)

                gamma = g_mean[i,j]*np.sqrt(1 + f_diff**2/(g_all[i] + g_all[j])**2)
                if gamma < gamma_m: gamma_m = gamma

    return gamma_m

freq_cutoff = [0.3,0.5,0.65]

num_modes = np.sum(f_ell1>freq_cutoff[0])

eps = np.zeros(num_modes)
for i in range(num_modes):
  eps[i] = gamma_min(f_ell1[-num_modes+i],n_g_ell1[-num_modes+i],1)/f_ell1[-num_modes+i]
  print(eps[i])
np.savetxt('eps_ell1_break.dat',eps)

num_modes = np.sum(f_ell2>freq_cutoff[1])

eps = np.zeros(num_modes)
for i in range(num_modes):
  eps[i] = gamma_min(f_ell2[-num_modes+i],n_g_ell2[-num_modes+i],2)/f_ell2[-num_modes+i]
  print(eps[i])
np.savetxt('eps_ell2_break.dat',eps)

num_modes = np.sum(f_ell3>freq_cutoff[2])

eps = np.zeros(num_modes)
for i in range(num_modes):
  eps[i] = gamma_min(f_ell3[-num_modes+i],n_g_ell3[-num_modes+i],3)/f_ell3[-num_modes+i]
  print(eps[i])
np.savetxt('eps_ell3_break.dat',eps)


