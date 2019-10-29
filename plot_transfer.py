
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import publication_settings

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

data = np.loadtxt('L_GYRE.dat')

freq = data[0,:]
transfer_ell1 = data[1,:]*np.pi/2
transfer_ell2 = data[2,:]*np.pi/2
transfer_ell3 = data[3,:]*np.pi/2

# making legend
plot_axes.loglog(freq, transfer_ell1)
plot_axes.loglog(freq, transfer_ell2)
plot_axes.loglog(freq, transfer_ell3)

lw = 2

reverse_order = False
if reverse_order:
  plot_axes.loglog(freq, transfer_ell3, label=r'$\ell=3$', color='DarkGoldenrod', linewidth=lw)
  plot_axes.loglog(freq, transfer_ell2, label=r'$\ell=2$', color='FireBrick', linewidth=lw)
  plot_axes.loglog(freq, transfer_ell1, label=r'$\ell=1$', color='MidnightBlue', linewidth=lw)
else:
  plot_axes.loglog(freq, transfer_ell1, label=r'$\ell=1$', color='MidnightBlue', linewidth=lw)
  plot_axes.loglog(freq, transfer_ell2, label=r'$\ell=2$', color='FireBrick', linewidth=lw)
  plot_axes.loglog(freq, transfer_ell3, label=r'$\ell=3$', color='DarkGoldenrod', linewidth=lw)

lg = plot_axes.legend(loc='upper left')
lg.draw_frame(False)

plot_axes.set_ylim([3e0,3e7])
plot_axes.set_xlim([4e-2,2.5])
plot_axes.set_xlabel(r'$f \ ({\rm d}^{-1})$')
plot_axes.set_ylabel(r'$T(f)$')

#plt.savefig('transfer.png',dpi=300)
plt.savefig('transfer.eps',dpi=300)

