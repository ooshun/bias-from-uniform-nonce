# Original files
- The implementation in this folder is largely based on the code base of Aranha, Novaes, Takahashi, Tibouchiand Yarom retrieved from [https://github.com/security-kouza/new-bleichenbacher-records](https://github.com/akiratk0355/ladderleak-attack-ecdsa)
- And I picked up some files from [https://github.com/security-kouza/new-bleichenbacher-records](https://github.com/security-kouza/new-bleichenbacher-records) to run the Makefile.


# Changed files
- I add some lines in `attack_mpi.cpp`, `mocksig.cpp`, `reduction.cpp`, `siggen_mpi.cpp` and their header files.

# How to run
- The setup procedure is the same as for [https://github.com/security-kouza/new-bleichenbacher-records](https://github.com/security-kouza/new-bleichenbacher-records).
- Generate HNP pair by executing `sh sign.sh`.
- Then, recover the secret key by executing `sh attack_known.sh`.
- For information on the command line arguments of `attack_mpi` and `siggen_mpi`, please refer to their respective help or in `sign.sh` and `attack_known.sh`.
