
from mpi4py import MPI

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

import numpy as np
import h5py
import os

f_list = np.linspace(-6.3,-5,1000)
f_list = 10**f_list
f_list_full = f_list

L_rms = []
f_list = []
for i,f in enumerate(f_list_full[rank::size]):
  output_folder = 'snapshots_damping_log10om_%.4f' %(np.log10(f))
  output_folder = output_folder.replace(".","p")
  output_folder = output_folder.replace("-","m")
  if os.path.exists('done/%s' %output_folder):
    f_list.append(f)
    filename = '%s/%s_s1/%s_s1_p0.h5' %(output_folder,output_folder,output_folder)
    file = h5py.File(filename)
    r = np.array(file['scales/r/1.0'])
    t = np.array(file['scales/sim_time'])
    i_r = np.argmin(np.abs(r-0.85))
    if len(t) > 2000:
      L = np.array(file['tasks/L'][-int(2/3*len(t)):,i_r])
    else:
      L = np.array(file['tasks/L'][-1000:,i_r])
    file.close()
    L_rms.append(np.sqrt(np.mean(L**2)))

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

  np.savetxt('Lrms_0p85.dat',data)

