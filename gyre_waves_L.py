
import numpy as np
import tomso as tomso
from tomso import gyre
import glob
import time
import matplotlib.pyplot as plt
from scipy import linalg

# load modes

def read_modes(file_bases):
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
  L_top = L[:,-1]

  return freq,omega,x,ur,uh,L,L_top,rho 

def calculate_L(bases,ell,om_list):
  freq, omega, x, ur, uh, L, L_top, rho = read_modes(bases)

  def IP(ur_1,ur_2,uh_1,uh_2):
    dx = np.gradient(x)
    return np.sum(dx*4*np.pi*x**2*rho*(np.conj(ur_1)*ur_2+ell*(ell+1)*np.conj(uh_1)*uh_2),axis=-1)
  
  IP_matrix = np.zeros((len(ur),len(ur)),dtype=np.complex128)
  for i in range(len(ur)):
    if i % 10 == 0: print(i)
    for j in range(len(ur)):
      IP_matrix[i,j] = IP(ur[i],ur[j],uh[i],uh[j])
  
  IP_inv = linalg.inv(IP_matrix)
  
  ur_dual = np.conj(IP_inv)@ur
  uh_dual = np.conj(IP_inv)@uh

  dx = np.gradient(x)
  L_top_list = []
  for i in range(50):
    bottom_point = 0.25+1e-4*i*4
    print(bottom_point)
    delta = 0*x
    RCB_index = np.argmin(np.abs(x - bottom_point))
    delta[RCB_index] = 1./dx[RCB_index]
    kperp = np.sqrt(ell*(ell+1))/x[RCB_index]
    L_top_om = np.abs(np.sum( IP(ur_dual,0*x,uh_dual,delta)[:,None]/(ell*(ell+1))*L_top[:,None]*om_list[None,:]/kperp/( om_list[None,:] - 1j*freq[:,None] ), axis=0 ))
    L_top_list.append(L_top_om)
  L_top_list = np.mean(np.array(L_top_list),axis=0)

  return L_top_list

def generate_om(ell):

  if ell == 1:
    om_list1 = np.linspace(-0.3333,0.666666,1000)
    om_list2 = np.linspace(0.666666,1.,10000)
    om_list3 = np.linspace(1,1.3333333,10000)
    om_list = np.concatenate([om_list1,om_list2,om_list3])
    om_list = 10**om_list
  
    om_start = [6.69 , 7.357, 8.182, 9.279, 10.7845, 12.8926]
    om_end   = [6.695, 7.363, 8.186, 9.282, 10.7857, 12.894 ]
    for om_s,om_e in zip(om_start,om_end):
      om_list = np.concatenate([om_list,np.linspace(om_s,om_e,1000)])
  
    om_start = [15.9194, 20.4268]
    om_end   = [15.9212, 20.4304]
    for om_s,om_e in zip(om_start,om_end):
      om_list = np.concatenate([om_list,np.linspace(om_s,om_e,10000)])
  
    sort = np.argsort(om_list)
    om_list = om_list[sort]
  elif ell == 2:
    om_list1 = np.linspace(-0.3333,0.666666,1000)
    om_list2 = np.linspace(0.666666,1.33333,10000)
    om_list = np.concatenate([om_list1,om_list2])
    om_list = 10**om_list
  
    om_start = [10.57773148, 11.5462963 , 12.68912037, 14.090625  , 15.95358796, 18.49513889]
    om_end   = [10.58449074, 11.54976852, 12.6912037 , 14.09537037, 15.95740741, 18.50081019]
    for om_s,om_e in zip(om_start,om_end):
      om_list = np.concatenate([om_list,np.linspace(om_s,om_e,1000)])
  
    sort = np.argsort(om_list)
    om_list = om_list[sort]
  elif ell == 3:
    om_list1 = np.linspace(-0.3333,0.666666,1000)
    om_list2 = np.linspace(0.666666,1.33333,10000)
    om_list = np.concatenate([om_list1,om_list2])
    om_list = 10**om_list
  
    om_start = [ 14.89074074, 16.23831019, 17.82523148, 19.76342593]
    om_end   = [ 14.89814815, 16.24641204, 17.83090278, 19.7693287 ]
    for om_s,om_e in zip(om_start,om_end):
      om_list = np.concatenate([om_list,np.linspace(om_s,om_e,1000)])
  
    sort = np.argsort(om_list)
    om_list = om_list[sort]
  return om_list*1e-6*24*60*60

base1 = '../gyre_igw/modes_10/10XC066_ell1.'
om_list1 = generate_om(1)

base2 = '../gyre_igw/modes_10/10XC066_ell2.'
om_list2 = generate_om(2)

base3 = '../gyre_igw/modes_10/10XC066_ell3.'
om_list3 = generate_om(3)

om_list = np.concatenate([om_list1,om_list2,om_list3])

om_list = np.unique(om_list)

L_top1 = calculate_L([base1],1,om_list)
L_top2 = calculate_L([base2],2,om_list)
L_top3 = calculate_L([base3],3,om_list)

data = np.vstack([om_list,L_top1,L_top2,L_top3])
np.savetxt('L_GYRE.dat',data)

