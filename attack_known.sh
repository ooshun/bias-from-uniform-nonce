start_time=`date +%s`
# 131-bit known 1 bit
LD_LIBRARY_PATH=./lib:/lib OMP_NUM_THREADS=48 mpirun -np 24 -x OMP_NUM_THREADS --map-by ppr:2:core --report-bindings --display-map ./attack_mpi --in data/ecdsa131_k1_2_a18_b1_f0 --test131 --red --a-vec 21 19 --v-vec 13 15 --n-vec 56 47 --m-vec 25 21 24 --dir red131_k1_2_2_21_19_13_15_56_47 --out 21_19_13_15_56_47 > red131_k1_2_2_21_19_13_15_56_47/red_log.txt
LD_LIBRARY_PATH=./lib:/lib OMP_NUM_THREADS=4 mpirun -np 2 -x OMP_NUM_THREADS --map-by ppr:2:core --report-bindings --display-map ./attack_mpi --in red131_k1_2_2_21_19_13_15_56_47/21_19_13_15_56_47_round-2 --test131 --a-vec 23 19 --v-vec 10 18 --n-vec 56 47 --m-vec 25 21 26.848 --fft-mpi --fft-mpi-batch-uint32_t 125316737 > red131_k1_2_2_21_19_13_15_56_47/fft_log.txt


end_time=`date +%s`
 
run_time=$((end_time - start_time))
 
echo $run_time
