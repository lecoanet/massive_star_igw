
import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools
import mesa_reader as mr
import h5py
import matplotlib.pyplot as plt
from scipy import interpolate

import logging
logger = logging.getLogger(__name__)
for h in logging.root.handlers:
  h.setLevel("DEBUG")

# Parameters
ell = 1

mesa_file = '../gyre_igw/10/LOGS/profile1.data.GYRE'

# load in data
file = open(mesa_file,'r')
lines = file.readlines()

header = lines[0].split()
n_r = int(header[0])
M_star = np.float(header[1])
R_star = np.float(header[2])
L_star = np.float(header[3])

r = np.zeros(n_r)
L_r = np.zeros(n_r)
M_r = np.zeros(n_r)
P = np.zeros(n_r)
T = np.zeros(n_r)
rho = np.zeros(n_r)
nabla = np.zeros(n_r)
N2 = np.zeros(n_r)
Gamma_1 = np.zeros(n_r)
nabla_ad = np.zeros(n_r)
delta = np.zeros(n_r)

for i in range(1,n_r+1):
  line = lines[i].split()
  r[i-1] = np.float(line[1])
  M_r[i-1] = np.float(line[2])
  L_r[i-1] = np.float(line[3])
  P[i-1] = np.float(line[4])
  T[i-1] = np.float(line[5])
  rho[i-1] = np.float(line[6])
  nabla[i-1] = np.float(line[7])
  N2[i-1] = np.float(line[8])
  Gamma_1[i-1] = np.float(line[9])
  nabla_ad[i-1] = np.float(line[10])
  delta[i-1] = np.float(line[11])

G = 6.67428e-8

c_P = P*delta/(rho*T*nabla_ad)
g = np.zeros(n_r)
g[1:] = G*M_r[1:]/r[1:]**2
g[0] = 0
dlogpdlogr = -g*rho*r/P
dlogTdlogr = dlogpdlogr*nabla

dsdr = N2/g
dsdr[0] = 0

i_RCB = np.argmax(N2>1e-50)
i_top = np.argmax(N2[i_RCB+1:]<1e-50) + i_RCB

r_bot = r[i_RCB]/r[-1]
r_top = r[i_top]/r[-1]-0.001

logger.info(r_bot)
logger.info(r_top)

N = 128

r_int1 = 0.5
r_int2 = 0.8
r_int3 = r_top
#r_int3 = 0.95

# Create bases and domain
r_basis1 = de.Chebyshev('r', 2*N, interval=(r_bot, r_int1) )
r_basis2 = de.Chebyshev('r', N, interval=(r_int1, r_int2) )
r_basis3 = de.Chebyshev('r', N, interval=(r_int2, r_int3) )
r_basis = de.Compound('r',[r_basis1,r_basis2,r_basis3])
domain = de.Domain([r_basis], grid_dtype=np.float64)

r_d = domain.grid(0)

def dedalus_interp(array):
    field = domain.new_field()
    interp = interpolate.interp1d(r/r[-1],array,kind='cubic')
    field['g'] = interp(r_d)
    return field

rho0 = dedalus_interp(rho)
g = dedalus_interp(g)
Gamma1 = dedalus_interp(Gamma_1)
P0 = dedalus_interp(P)
T0 = dedalus_interp(T)
nabla_ad = dedalus_interp(nabla_ad)
L0 = dedalus_interp(L_r)
cp = dedalus_interp(c_P)
nuT = dedalus_interp(delta)
dlogpdlogr = dedalus_interp(dlogpdlogr)
dlogTdlogr = dedalus_interp(dlogTdlogr)

rho0c = rho0['g'][0]
rho0['g'] /= rho0c
P0c = P0['g'][0]
P0['g'] /= P0c
gc = g['g'][0]
g['g'] /= gc
T0c = T0['g'][0]
T0['g'] /= T0c
L0c = L0['g'][0]
L0['g'] /= L0c
cpc = cp['g'][0]
cp['g'] /= cpc

# this is actually a frequency normalization!
T_norm = np.sqrt(P0c/rho0c)/R_star
logger.info(T_norm)
prefactor = 1/T0c/np.sqrt(P0c*rho0c)/cpc*L0c/R_star**2

cs2 = domain.new_field()
cs2['g'] = Gamma1['g']*P0['g']/rho0['g']

logrho = np.log(rho0['g'])
dlogrho0dr = domain.new_field()
dlogrho0dr['g'] = dlogpdlogr['g']/r_d/Gamma1['g'] - nuT['g']*(dlogTdlogr['g']/r_d-nabla_ad['g']*dlogpdlogr['g']/r_d)

dp0dr_over_rho = domain.new_field()
dp0dr_over_rho['g'] = dlogpdlogr['g']/r_d*P0['g']/rho0['g']

dT0dr = domain.new_field()
dT0dr['g'] = dlogTdlogr['g']/r_d*T0['g']

#f_list = np.array([2e-7,4e-7,1e-6,2e-6,4e-6,1e-5])
f_list = [1.20e-6]

for f in f_list:
  om = f*(2*np.pi) # rad/sec
  om /= T_norm

  # period
  DT = 2*np.pi/om
  dt_scale = 4
  dt = DT/100*np.pi/3*dt_scale

  # p = p'/rho0; rho = rho'/rho0
  variables = ['ur','uh','p','rho','divu','T','L']
  problem = de.IVP(domain, variables=variables, ncc_cutoff=1e-5, ncc_relative_cutoff=1e-5)
  
  for variable in variables:
    problem.meta[variable]['r']['dirichlet'] = True
  
  problem.parameters['ell'] = ell
  problem.parameters['pi'] = np.pi
  problem.parameters['R_star'] = R_star
  problem.parameters['pc'] = P0c
  problem.parameters['p0'] = P0
  problem.parameters['dpdr_over_rho'] = dp0dr_over_rho
  problem.parameters['rhoc'] = rho0c
  problem.parameters['rho0'] = rho0
  problem.parameters['dlogrhodr'] = dlogrho0dr
  problem.parameters['T0'] = T0
  problem.parameters['dTdr'] = dT0dr
  problem.parameters['gc'] = gc
  problem.parameters['g'] = g
  problem.parameters['cs2'] = cs2
  problem.parameters['L0'] = L0
  problem.parameters['cp'] = cp
  problem.parameters['prefactor'] = prefactor
  problem.parameters['nabla_ad'] = nabla_ad
  problem.parameters['nuT'] = nuT
  problem.parameters['Gamma1'] = Gamma1
  problem.parameters['om'] = om
  problem.parameters['t0'] = dt*1000/dt_scale
  problem.parameters['t_ramp'] = 100*dt/dt_scale

  problem.add_equation("dt(ur) + dr(p) + p*dlogrhodr + gc*rhoc*R_star/pc*g*rho = 0")
  
  problem.add_equation("dt(uh) + p/r = 0")
  
  problem.add_equation("r**2*(divu - (1/r**2*dr(r**2*ur) - ell*(ell+1)/r*uh)) = 0")
  
  problem.add_equation("dt(rho) + ur*dlogrhodr + divu = 0")
  
  problem.add_equation("nuT*T - (nuT*nabla_ad + 1/Gamma1)*T0*rho0/p0*p + T0*rho = 0")
  
  problem.add_equation("L - L0*(dr(T)/dTdr + 3*T/T0 - rho) = 0")
  
  problem.add_equation("dt(p) + dpdr_over_rho*ur + cs2*divu"
                             " + prefactor*cs2*nuT/rho0/T0/cp/(4*pi*r**2)*( dr(L) + L0/dTdr*ell*(ell+1)/r**2*T ) = 0")
  
  problem.add_bc("left(ur) = sin(om*t)*(1+tanh( (t - t0)/t_ramp ) )/2")
  problem.add_bc("right(ur) = 0")
  problem.add_bc("left(rho - rho0/p0*p/Gamma1) = 0")
  problem.add_bc("right(rho - rho0/p0*p/Gamma1) = 0")
  
  # Build solver
  solver = problem.build_solver(de.timesteppers.RK443)
  logger.info('Solver built')
  
  # Integration parameters
  solver.stop_sim_time = np.inf
  solver.stop_wall_time = np.inf
  solver.stop_iteration = int(10000/dt_scale)
  
  # Analysis
  output_folder = 'snapshots_damping_log10om_%.2f' %(np.log10(f))
  output_folder = output_folder.replace(".","p")
  output_folder = output_folder.replace("-","m")
  print(output_folder)
  snapshots = solver.evaluator.add_file_handler(output_folder, iter = 5)
  snapshots.add_system(solver.state)
  
  # Flow properties
  flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
  flow.add_property("sqrt(ur*ur)", name='ur')

  L_max = 0
  iter_max = 0
  L = solver.state['L']
  
  # Main loop
  try:
      logger.info('Starting loop')
      start_time = time.time()
      while solver.ok:
          solver.step(dt)
          if (solver.iteration-1) % 10 == 0:
              logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
              logger.info('Max ur = {}'.format(flow.max('ur')))
              current_L_max = np.abs(L['g'][-1])
              if current_L_max > L_max*1.1:
                  L_max = current_L_max
                  iter_max = solver.iteration
                  logger.info('new L max: %e' %L_max)
                  if iter_max > solver.stop_iteration//3:
                      solver.stop_iteration = iter_max*3
                      logger.info(solver.stop_iteration)
  except:
      logger.error('Exception raised, triggering end of main loop.')
      raise
  finally:
      end_time = time.time()
      logger.info('Iterations: %i' %solver.iteration)
      logger.info('Sim end time: %f' %solver.sim_time)
      logger.info('Run time: %.2f sec' %(end_time-start_time))
      logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))

