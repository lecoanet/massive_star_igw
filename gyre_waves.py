
import numpy as np
import tomso as tomso
from tomso import gyre
import glob
import time
import matplotlib.pyplot as plt
from scipy import linalg

# load modes

def read_modes(file_bases, measurement_point=1):
  freq_list = []
  xir_list = []
  xih_list = []
  L_list = []
  omega_list = []
  file_list = [file for base in file_bases for file in glob.glob('%s*.txt' %base)]

  for i,filename in enumerate(file_list):
    if i % 10 == 0: print(filename)
    header, data_mode = tomso.gyre.load_summary(filename)
    freq_list.append(-1j*(header['Refreq'] + 1j*header['Imfreq']))
    omega_list.append(header['Reomega'] + 1j*header['Imomega'])
    xir_list.append(data_mode['Rexi_r'] + 1j*data_mode['Imxi_r'])
    xih_list.append(data_mode['Rexi_h'] + 1j*data_mode['Imxi_h'])
    L_list.append(data_mode['Relag_L'] + 1j*data_mode['Imlag_L'])
  
  header, data_mode = tomso.gyre.load_summary(glob.glob('%s*.txt' %file_bases[0])[0])
  rho = data_mode['rho']
  x = data_mode['x']

  freq = np.array(freq_list)*1e-6*24*60*60
  omega = np.array(omega_list)
  ur = np.array(xir_list)*1j*freq[:,None]
  uh = np.array(xih_list)*1j*freq[:,None]
  L = np.array(L_list)
  i_r = np.argmin(np.abs(x-measurement_point))
  L_top = L[:,i_r]

  return freq,omega,x,ur,uh,L,L_top,rho 

def calculate_duals(x,ur,uh,ell,IP):

  IP_matrix = np.zeros((len(ur),len(ur)),dtype=np.complex128)
  for i in range(len(ur)):
    if i % 10 == 0: print(i)
    for j in range(len(ur)):
      IP_matrix[i,j] = IP(ur[i],ur[j],uh[i],uh[j])
 
  IP_inv = linalg.inv(IP_matrix)
 
  ur_dual = np.conj(IP_inv)@ur
  uh_dual = np.conj(IP_inv)@uh

  return ur_dual, uh_dual

def calculate_L(x,ur_dual,uh_dual,L_top,freq,ell,om_list, IP, x_bot):

  dx = np.gradient(x)
  L_top_list = []
  for i in range(50):
    bottom_point = x_bot+1e-4*i*4
    if i % 10 == 0: print(bottom_point)
    delta = 0*x
    RCB_index = np.argmin(np.abs(x - bottom_point))
    delta[RCB_index] = 1./dx[RCB_index]
    kperp = np.sqrt(ell*(ell+1))/x[RCB_index]
    L_top_om = np.abs(np.sum( IP(ur_dual,0*x,uh_dual,delta)[:,None]/np.sqrt(ell*(ell+1))*L_top[:,None]*om_list[None,:]/kperp/( om_list[None,:] - 1j*freq[:,None] ), axis=0 ))
    L_top_list.append(L_top_om)
  L_top_list = np.mean(np.array(L_top_list),axis=0)

  return L_top_list

def calculate_L_profile(ell,mass,CX,x_bot = 0.25,base = None,measurement_point=1):
  if base == None:
    base = '../gyre_igw/modes_%i/%iXC%s_ell%i.' %(mass,mass,CX,ell)
  
  freq, omega, x, ur, uh, L, L_top, rho = read_modes([base],measurement_point=measurement_point)
  
  def IP(ur_1,ur_2,uh_1,uh_2):
    dx = np.gradient(x)
    return np.sum(dx*4*np.pi*x**2*rho*(np.conj(ur_1)*ur_2+ell*(ell+1)*np.conj(uh_1)*uh_2),axis=-1)
  
  ur_dual, uh_dual = calculate_duals(x,ur,uh,ell,IP)
  
  freq_min = np.min(-freq.imag)*1.5
  freq_max = np.max(-freq.imag)/1.5
  
  dfreq = (-freq.imag[1:]) - (-freq.imag[:-1])
  gamma = freq.real[1:]
  i_mode = np.argmin(dfreq - gamma)
  freq_mode = -freq[i_mode].imag
  i_mode = np.argmin(np.abs(freq_mode - (-freq.imag)))
  
  n_low_om = 1000
  n_high_om = 10000
  om_list1 = np.linspace(np.log10(freq_min),np.log10(freq_mode),n_low_om)
  om_list2 = np.linspace(np.log10(freq_mode),np.log10(freq_max),n_high_om)
  om_list = np.concatenate([om_list1,om_list2])
  om_list = 10**om_list
  
  L_top_list = calculate_L(x,ur_dual,uh_dual,L_top,freq,ell,om_list,IP,x_bot)
  
  def refine_peaks(om_list,L_top_list):
    i_peaks = []
    for i in range(len(om_list) - n_low_om - 1):
      j = n_low_om + i
      if (L_top_list[j]>L_top_list[j-1]) and (L_top_list[j]>L_top_list[j+1]):
        delta_m = np.abs(L_top_list[j]-L_top_list[j-1])/L_top_list[j]
        delta_p = np.abs(L_top_list[j]-L_top_list[j+1])/L_top_list[j]
        if delta_m > 0.05 or delta_p > 0.05:
          i_peaks.append(j)
  
    print(len(i_peaks))
  
    om_new = np.array([])  
    for i in i_peaks:
      om_low = om_list[i-1]
      om_high = om_list[i+1]
      om_new = np.concatenate([om_new,np.linspace(om_low,om_high,10)])
    
    L_top_new = calculate_L(x,ur_dual,uh_dual,L_top,freq,ell,om_new,IP,x_bot)
  
    om_list = np.concatenate([om_list,om_new])
    L_top_list = np.concatenate([L_top_list,L_top_new])
  
    om_list, sort = np.unique(om_list, return_index=True)
    L_top_list = L_top_list[sort]
  
    return om_list, L_top_list, len(i_peaks)
  
  peaks = 1
  while peaks > 0:
    om_list, L_top_list, peaks = refine_peaks(om_list,L_top_list)

  return om_list, L_top_list

for ell in [1]:
  for mass in [3,7,10,20]:
    for CX in ['066','033','000']:
      if mass == 20 and CX =='066': x_bot = 0.35
      else: x_bot = 0.25
      om, L_top = calculate_L_profile(ell,mass,CX,x_bot=x_bot)
      filename = 'L_%i_%s_ell%s.dat' %(mass,CX,ell)
      np.savetxt('data/'+filename,np.vstack([om,L_top]))

for ell in [2,3]:
  for mass in [10]:
    for CX in ['066']:
      om, L_top = calculate_L_profile(ell,mass,CX,x_bot=0.25)
      filename = 'L_%i_%s_ell%s.dat' %(mass,CX,ell)
      np.savetxt('data/'+filename,np.vstack([om,L_top]))

for ell in [1]:
  for mass in [10]:
    for CX in ['066']:
      om, L_top = calculate_L_profile(ell,mass,CX,x_bot=x_bot,measurement_point=0.85)
      filename = 'L_%i_%s_ell%s_0p85.dat' %(mass,CX,ell)
      np.savetxt(filename,np.vstack([om,L_top]))
      om, L_top = calculate_L_profile(ell,mass,CX,x_bot=x_bot,measurement_point=0.9755)
      filename = 'L_%i_%s_ell%s_0p98.dat' %(mass,CX,ell)
      np.savetxt(filename,np.vstack([om,L_top]))


