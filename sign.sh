# 131-bit
# LD_LIBRARY_PATH=./lib:/lib OMP_NUM_THREADS=1 mpirun -np 1 ./siggen_mpi --leak 2 --msbs 3 --test131 --filter 0 --out data/ecdsa131_2_a24_b2_f0

# LD_LIBRARY_PATH=./lib:/lib OMP_NUM_THREADS=1 mpirun -np 1 ./siggen_mpi --leak 2 --msbs 3 --error-rate 0.111434 --test131 --filter 0 --out data/ecdsa131_2_a24_b2_f0_e01114_2

# LD_LIBRARY_PATH=./lib:/lib OMP_NUM_THREADS=1 mpirun -np 1 ./siggen_mpi --leak 2 --msbs 3 --error-rate 0.1 --test131 --filter 0 --out data/ecdsa131_2_a22_b2_f0_e01

# 131-bit known 1 bit
LD_LIBRARY_PATH=./lib:/lib OMP_NUM_THREADS=1 mpirun -np 1 ./siggen_mpi --leak 1 --msbs 1 --test131 --filter 0 --out data/ecdsa131_k1_2_a21_b1_f0


# 131-bit known 2 bit with error
# LD_LIBRARY_PATH=./lib:/lib OMP_NUM_THREADS=1 mpirun -np 1 ./siggen_mpi --leak 2 --msbs 3 --error-rate 0.111434 --test131 --filter 0 --out data/ecdsa131_k2-2_2_a21_b2_f0_e01114


# # 131-bit known 3 bit
# LD_LIBRARY_PATH=./lib:/lib OMP_NUM_THREADS=1 mpirun -np 1 ./siggen_mpi --leak 3 --msbs 3 --test131 --filter 0 --out data/ecdsa131_k3_2_a14_b3_f0_07


# 162-bit
# LD_LIBRARY_PATH=./lib:/lib OMP_NUM_THREADS=1 mpirun -np 1 ./siggen_mpi --leak 1 --msbs 1 --prime163r1 --filter 0 --out data/ecdsa163_a26_b1_f0

# LD_LIBRARY_PATH=./lib:/lib OMP_NUM_THREADS=1 mpirun -np 1 ./siggen_mpi --leak 2 --msbs 3 --error-rate 0.1 --prime163r1 --filter 0 --out data/ecdsa163_2_a22_b1_f0
