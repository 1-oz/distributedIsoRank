from mpi4py import MPI
import sys
print(sys.executable)
print(sys.version)
comm = MPI.COMM_WORLD
print(f"Hello from rank {comm.Get_rank()} of {comm.Get_size()}")

