
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import publication_settings

matplotlib.rcParams.update(publication_settings.params)

t_mar, b_mar, l_mar, r_mar = (0.15, 0.25, 0.29, 0.02)
h_plot, w_plot = (1, 1/publication_settings.golden_mean)
w_pad = 0.35

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

# dedalus data
T_norm = 0.00018418960703617103
om_full, L_top_full = np.loadtxt('Lrms_time_full.dat')
om_0p85_full, L_0p85_full = np.loadtxt('Lrms_0p85.dat')
om_fine, L_top_fine = np.loadtxt('Lrms_time_fine.dat')
om_0p85_fine, L_0p85_fine = np.loadtxt('Lrms_0p85_fine.dat')
om_long, L_top_long = np.loadtxt('Lrms_time_2hour.dat')
om_0p85_long, L_0p85_long = np.loadtxt('Lrms_0p85_2hour.dat')

om_min = np.min(om_long)
om_d = np.concatenate([om_full[om_full<om_min],om_fine,om_long])
L_top_d = np.concatenate([L_top_full[om_full<om_min],L_top_fine,L_top_long])
om_0p85_d = np.concatenate([om_0p85_full[om_0p85_full<om_min],om_0p85_fine,om_0p85_long])
L_0p85_d = np.concatenate([L_0p85_full[om_0p85_full<om_min],L_0p85_fine,L_0p85_long])

sort = np.argsort(om_d)
om_d = om_d[sort]
L_top_d = L_top_d[sort]

sort = np.argsort(om_0p85_d)
om_0p85_d = om_0p85_d[sort]
L_0p85_d = L_0p85_d[sort]

# remove nan & weird data point
i_nan = np.argmax(np.isnan(L_top_d))
i_weird = 200 + np.argmin(L_top_d[200:])
for i in [i_weird,i_nan]:
  om_d = np.delete(om_d,i)
  L_top_d = np.delete(L_top_d,i)

om_d *= 24*60*60 # d^-1
L_top_d *= T_norm*24*60*60 # d^-1

om_0p85_d *= 24*60*60 # d^-1
L_0p85_d *= T_norm*24*60*60 # d^-1

om_g, L_top_g = np.loadtxt('L_10_066_ell1_0p98.dat')
om_0p85_g, L_0p85_g = np.loadtxt('L_10_066_ell1_0p85.dat')

L_top_g *= np.pi/2
L_0p85_g *= np.pi/2

lw = 1

scale_factor = 6

plot_axes[0].loglog(om_0p85_d,L_0p85_d/scale_factor,label=r'Dedalus',color='MidnightBlue',linewidth=lw)
plot_axes[0].loglog(om_0p85_g,L_0p85_g,label=r'GYRE',color='Firebrick',linewidth=lw)

plot_axes[1].loglog(om_d,L_top_d/scale_factor,label=r'Dedalus',color='MidnightBlue',linewidth=lw)
plot_axes[1].loglog(om_g,L_top_g,label=r'GYRE',color='Firebrick',linewidth=lw)

r_L = [r'$r=0.85$',r'$r\approx 0.976$']

for i,plot_axis in enumerate(plot_axes):
  plot_axis.set_ylim([3e-1,1e8])
  plot_axis.set_xlim([4e-2,2.5])
  lg = plot_axis.legend(loc='upper left')
  lg.draw_frame(False)

  plot_axis.set_xlabel(r'$f \ ({\rm d}^{-1})$')

  plot_axis.text(0.5,1.05,r_L[i],horizontalalignment='center',transform=plot_axis.transAxes,fontsize=10)

plot_axes[0].set_ylabel(r'$T(f)$')
plot_axes[1].set_ylabel(r'$T(f)$')

plt.savefig('comparison.png',dpi=300)
#plt.savefig('comparison.eps')

