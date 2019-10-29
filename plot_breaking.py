
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import publication_settings

matplotlib.rcParams.update(publication_settings.params)

t_mar, b_mar, l_mar, r_mar = (0.05, 0.22, 0.29, 0.05)
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

freq_full = data_234517653[0]
spectrum_star = data_234517653[1]

colors = ['MidnightBlue','FireBrick','DarkGoldenrod']

freq_cutoff = [0.3,0.5,0.65]

for i in range(3):
  data = np.loadtxt('L_nl_ell%i.dat' %(i+1))
  plot_axes.scatter(data[0],data[1]*1e3,marker='*',color = colors[i],label = r'$\ell=%i$' %(i+1),s=20,zorder=5-i)

lw = 1
plot_axes.loglog(freq_full, spectrum_star,linewidth=0.5,color='grey',label='EPIC 234517653',zorder=-1)

lg = plot_axes.legend(loc='upper left',scatterpoints = 1)
lg.draw_frame(False)

plot_axes.set_ylim([1e-3,3e1])
plot_axes.set_xlim([4e-2,10])
plot_axes.set_xlabel(r'$f \ ({\rm d}^{-1})$')
plot_axes.set_ylabel(r'$\delta L \ ({\rm mmag})$')

plt.savefig('break.png',dpi=300)
#plt.savefig('break.eps')

