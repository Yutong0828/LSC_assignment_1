from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as sts

def sim_health_parallel(n_runs=1000):
  #Get rank of process and overall size of communicator:
  comm=MPI.COMM_WORLD
  rank=comm.Get_rank()
  size=comm.Get_size()
  print("size:",size)

  #Start time:
  t0=time.time()

  #Evenly distribute number of simulation runs across processes
  N=int(n_runs/size)
  print(N)

  # Simulate on each MPI Process and specify as a NumPy Array
  rho=0.5
  mu=3.0
  sigma=1.0
  z_0=mu

  T=int(4160)
  np.random.seed(rank)
  eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, N))

  z_mat = np.zeros((T, N)) #results saved here

  for s_ind in range(N):
    z_tm1=z_0
    for t_ind in range(T):
      e_t=eps_mat[t_ind,s_ind]
      z_t=rho*z_tm1+(1-rho)*mu+e_t
      z_mat[t_ind,s_ind]=z_t
      z_tm1=z_t

  
  #Gather all simulation arrays 
  z_mat_all=None
  if rank==0:
    z_mat_all=np.empty([T,n_runs],dtype="float")
  comm.Gather(sendbuf=z_mat,recvbuf=z_mat_all,root=0)

  #Print/plot simulation results on rank 0
  if rank==0:
    time_elapsed=time.time()-t0

    #Print time elapsed
    print("With %d cores, the time cost is %f"%(size,time_elapsed))
  
  return

def main():
  sim_health_parallel(n_runs=1000)

if __name__ == '__main__':
  main()
