import time
start_time=time.time()

import numpy as np
import scipy.stats as sts

# Set model parameters
rho = 0.5
mu = 3.0
sigma=1.0
z_0=mu

# Set simulation parameters, draw all idiosyncratic random shocks,
# and create empty containers

S = 1000 # Set the number of lives to simulate
T = int(4160) # Set the number of periods for each simulation
np.random.seed(25)
eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
#z_mat = np.zeros((T, S))


#ahead of time Numba
from numba.pycc import CC

cc=CC('test_aot')
@cc.export('simu_aot','f8[:,:](i8,i8)')
def simu_aot(S,T):
  z_mat = np.zeros((T, S))
  for s_ind in range(S):
    z_tm1=z_0
    for t_ind in range(T):
      e_t=eps_mat[t_ind,s_ind]
      z_t=rho*z_tm1+(1-rho)*mu+e_t
      z_mat[t_ind,s_ind]=z_t
      z_tm1=z_t
  return z_mat

cc.compile()

import test_aot
test_aot.simu_aot(S,T)

end_time=time.time()

print("time cost:",end_time-start_time)