
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import LogLocator
import matplotlib.pyplot as plt
import publication_settings

matplotlib.rcParams.update(publication_settings.params)

t_mar, b_mar, l_mar, r_mar = (0.15, 0.37, 0.45, 0.35)
h_plot, w_plot = (1, 1/publication_settings.golden_mean)
w_pad = 0.05
h_pad = 0.05

h_total = t_mar + 4*h_plot + 3*h_pad + b_mar
w_total = l_mar + 3*w_plot + 2*w_pad + r_mar

width = 7.1
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

plot_axes = []
for i in range(3):
  for j in range(4):
    left = (l_mar + i*w_plot + i*w_pad) / w_total
    bottom = 1 - (t_mar + (j+1)*h_plot + j*h_pad ) / h_total
    width = w_plot / w_total
    height = h_plot / h_total
    plot_axes.append(fig.add_axes([left, bottom, width, height]))

text_top = ['ZAMS','mid','TAMS']
text_side = [r'$3M_\odot$',r'$7M_\odot$',r'$10M_\odot$',r'$20M_\odot$']

for i,CX in enumerate(['066','033','000']):
  for j,mass in enumerate([3,7,10,20]):

    index = 4*i + j
    plot_axis = plot_axes[index]

    om_ell1, transfer_ell1 = np.loadtxt('L_%i_%s_ell1.dat' %(mass,CX))
    transfer_ell1 *= np.pi/2

    lw1 = 0.5
    reverse = True
    plot_axis.loglog(om_ell1, transfer_ell1, label=r'$\ell=1$', color='MidnightBlue', linewidth=lw1)

    if CX == '066':
        plot_axis.set_xlim([3e-2,7e0])
    elif CX == '033':
        plot_axis.set_xlim([2e-2,7e0])
    elif CX == '000':
        plot_axis.set_xlim([2e-2,6e0])

    if mass == 3:
        plot_axis.set_ylim([1e1,1e10])
    elif mass == 7:
        plot_axis.set_ylim([1e1,3e8])
    elif mass == 10:
        plot_axis.set_ylim([1e0,1e9])
    elif mass == 20:
        plot_axis.set_ylim([1e1,3e7])

    if j == 0:
      plot_axis.text(0.5,1.05,text_top[i],horizontalalignment='center',transform=plot_axis.transAxes,fontsize=10)

    if i == 2:
      plot_axis.text(1.12,0.5,text_side[j],horizontalalignment='center',verticalalignment='center',transform=plot_axis.transAxes,fontsize=10)

    if j < 3:
      plt.setp(plot_axis.get_xticklabels(), visible=False)
    else:
      plot_axis.set_xlabel(r'$f \, ({\rm d}^{-1})$')
    if i > 0:
      plt.setp(plot_axis.get_yticklabels(), visible=False)
    else:
      plot_axis.set_ylabel(r'$T(f)$')
      plt.setp(plot_axis.get_yticklabels()[-2:], visible=False)

#plt.savefig('transfer_grid.png',dpi=300)
plt.savefig('transfer_grid.eps')

