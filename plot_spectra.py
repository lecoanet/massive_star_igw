
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import publication_settings
import csv

matplotlib.rcParams.update(publication_settings.params)

t_mar, b_mar, l_mar, r_mar = (0.05, 0.2, 0.29, 0.02)
h_plot, w_plot = (1, 1/publication_settings.golden_mean)
w_pad = 0.03

h_total = t_mar + h_plot + b_mar
w_total = l_mar + w_plot + r_mar

width = 4.3
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

left = (l_mar) / w_total
bottom = 1 - (t_mar + h_plot ) / h_total
width = w_plot / w_total
height = h_plot / h_total
plot_axes = fig.add_axes([left, bottom, width, height])

data_234517653 = np.loadtxt('EPIC234517653.dat')
data = np.loadtxt('spectra.dat')
freq = data[0]
spectrum_R = data[1]
spectrum_LQ = data[2]

freq_obs = data_234517653[0]
spectrum_obs = data_234517653[1]

spectrum_R *= 5e-8
spectrum_LQ *= 2e-11

lw = 1
plot_axes.loglog(freq_obs, spectrum_obs,linewidth=1,color='k',label='EPIC 234517653')
plot_axes.loglog(freq, spectrum_R*1e3,linewidth=lw,color='Firebrick',label='R spectrum')
plot_axes.loglog(freq, spectrum_LQ*1e3,linewidth=lw,color='DarkGoldenrod',lw=1.5,label='LQ spectrum')

lg = plot_axes.legend(loc='upper left')
lg.draw_frame(False)

plot_axes.set_ylim([3e-4,3e2])
plot_axes.set_xlim([4e-2,3.5])
plot_axes.set_xlabel(r'$f \ ({\rm d}^{-1})$')
plot_axes.set_ylabel(r'$\delta L \ ({\rm scaled})$')

plt.savefig('spectra.png',dpi=300)
#plt.savefig('spectra.eps')

