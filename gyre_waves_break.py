
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
  xih_list = []
  L_list = []
  omega_list = []
  file_list = [file for base in file_bases for file in glob.glob('%s*.txt' %base)]

  for i,filename in enumerate(file_list):
    if i % 10 == 0: print(filename)
    header, data_mode = tomso.gyre.load_summary(filename)
    freq_list.append(-1j*(header['Refreq'] + 1j*header['Imfreq']))
    omega_list.append(header['Reomega'] + 1j*header['Imomega'])
    xih_list.append(data_mode['Rexi_h'] + 1j*data_mode['Imxi_h'])
    L_list.append(data_mode['Relag_L'] + 1j*data_mode['Imlag_L'])
  
  header, data_mode = tomso.gyre.load_summary(glob.glob('%s*.txt' %file_bases[0])[0])
  rho = data_mode['rho']
  x = data_mode['x']

  freq = np.array(freq_list)*1e-6*24*60*60
  omega = np.array(omega_list)
  xih = np.array(xih_list)
  L = np.array(L_list)
  L_top = L[:,-1]

  return freq,omega,x,xih,L_top 

base_list = ['../gyre_igw/modes_10/10XC066_ell1.','../gyre_igw/modes_10/10XC066_ell2.','../gyre_igw/modes_10/10XC066_ell3.']

b_ell = np.array([0.708,0.325,0.0626])

for i,base in enumerate(base_list):
  freq, omega, x, xih, L_top = read_modes([base])
  
  eps = np.max(np.abs(xih/x)[:,1000:],axis=1)
  
  eps_target = np.loadtxt('eps_ell%i_break.dat' %(i+1))
  num = len(eps_target)
  eps = eps[-num:]
  freq = freq[-num:]
  L_top = L_top[-num:]
  L = np.abs(L_top)*b_ell[i]*eps_target/eps
  data = np.vstack([freq,L])
  np.savetxt('L_nl_ell%i.dat' %(i+1),data)

