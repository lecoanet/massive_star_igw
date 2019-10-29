
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import publication_settings
import csv

matplotlib.rcParams.update(publication_settings.params)

t_mar, b_mar, l_mar, r_mar = (0.1, 0.25, 0.33, 0.02)
h_plot, w_plot = (1, 1/publication_settings.golden_mean)
w_pad = 0.1

h_total = t_mar + h_plot + b_mar
w_total = l_mar + 2*w_plot + w_pad + r_mar

width = 7.1
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

plot_axes = []
for i in range(2):
  left = (l_mar + i*w_plot + i*w_pad) / w_total
  bottom = 1 - (t_mar + h_plot) / h_total
  width = w_plot / w_total
  height = h_plot / h_total
  plot_axes.append(fig.add_axes([left, bottom, width, height]))

data_234517653 = np.loadtxt('EPIC234517653.dat')

text_top = [r'$\Omega = 0.3 \, {\rm d}^{-1}$', r'$\Omega = 0.5 \, {\rm d}^{-1}$']
f = [0.6,1]

for i in range(2):
  if i == 0: data = np.loadtxt('spectra_Om0p3.dat')
  elif i == 1: data = np.loadtxt('spectra_Om0p5.dat')

  freq = data[0]
  spectrum_R = data[1]
  spectrum_LQ = data[2]
  
  freq_obs = data_234517653[0]
  spectrum_obs = data_234517653[1]
  
  spectrum_R *= 5e-8
  spectrum_LQ *= 2e-11
  
  lw = 1
  plot_axes[i].axvspan(4e-2, f[i], color=(0.9,0.9,0.9))
  plot_axes[i].loglog(freq_obs, spectrum_obs,linewidth=1,color='k',label='EPIC 234517653')
  plot_axes[i].loglog(freq, spectrum_R*1e3,linewidth=lw,color='Firebrick',label='R spectrum')
  plot_axes[i].loglog(freq, spectrum_LQ*1e3,linewidth=lw,color='DarkGoldenrod',lw=1.5,label='LQ spectrum')
  
  lg = plot_axes[i].legend(loc='upper left')
  lg.draw_frame(False)
  
  plot_axes[i].set_ylim([3e-4,3e2])
  plot_axes[i].set_xlim([4e-2,3.5])
  plot_axes[i].set_xlabel(r'$f \ ({\rm d}^{-1})$')
  if i == 0:
    plot_axes[i].set_ylabel(r'$\delta L \ ({\rm scaled})$')
  if i == 1:
    plt.setp(plot_axes[i].get_yticklabels(), visible=False)
  plot_axes[i].text(0.5,1.03,text_top[i],horizontalalignment='center',transform=plot_axes[i].transAxes,fontsize=10)

plt.savefig('spectra_rotation.png',dpi=300)
#plt.savefig('spectra_rotation.eps')

