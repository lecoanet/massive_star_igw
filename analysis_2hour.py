
import glob
from mpi4py import MPI

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

import numpy as np
import h5py
import os

f_list = np.linspace(-6.3,-5,1000)
f_list = 10**f_list
f_list_full = f_list

f_list_full = f_list_full[680:]

L_rms = []
f_list = []
for i,f in enumerate(f_list_full[rank::size]):
  output_folder = 'snapshots_damping_log10om_%.4f' %(np.log10(f))
  output_folder = output_folder.replace(".","p")
  output_folder = output_folder.replace("-","m")
  print(output_folder)
  t = np.array([])
  L = np.array([])
  for filename in glob.glob('%s/*/*.h5' %output_folder):
    file = h5py.File(filename)
    t = np.concatenate([t,np.array(file['scales/sim_time'])])
    L = np.concatenate([L,np.array(file['tasks/L'][:,-1])])
    file.close()

  t_order = np.argsort(t)
  t = t[t_order]
  L = L[t_order]

  if len(t) > 2000:
    L = L[-int(2/3*len(t)):]
  else:
    L = L[-1000:]
  L_rms.append(np.sqrt(np.mean(L**2)))
  f_list.append(f)

L_rms = np.array(L_rms)
f_list = np.array(f_list)

data = np.vstack((f_list,L_rms))

f_list = MPI.COMM_WORLD.gather(f_list,root=0)
L_rms_list = MPI.COMM_WORLD.gather(L_rms,root=0)

if rank==0:
  f = [freq for sub_list in f_list for freq in sub_list]
  L_rms = [L for sub_list in L_rms_list for L in sub_list]
  f = np.array(f)
  L_rms = np.array(L_rms)
  order = np.argsort(f)
  f = f[order]
  L_rms = L_rms[order]

  data = np.vstack((f,L_rms))

  np.savetxt('Lrms_time_2hour.dat',data)

