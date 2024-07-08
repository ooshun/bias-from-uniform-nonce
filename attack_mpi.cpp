#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <queue>
#include <bitset>
#include <numeric>
#include <algorithm>

#include <gmpxx.h>
#include <boost/mpi.hpp>
#include <boost/multiprecision/gmp.hpp>
#include <boost/serialization/string.hpp>
#include <boost/program_options.hpp>
#include <fftw3.h>

#include "mocksig.h"
#include "reduction.h"
#include "fft.h"

// #define SAVE_BIAS // save bias as csv
// #define FFT_h // calculate and use FFT_h

namespace mpi = boost::mpi;
namespace po = boost::program_options;

int main(int argc, char **argv)
{
	/* MPI */
	mpi::environment env;
	mpi::communicator world;
	const int master = 0;
	int myrank, worldsize;
	myrank = world.rank();
	worldsize = world.size();

	/* parse command line arguments */
	po::options_description desc("Allowed options");
	desc.add_options()("help", "help message")("verbose", "verbose logging")("test90", "run in test mode")("test90-mpi", "run in test mode")("test10", "run in test mode")("test60", "run in test mode")("test131", "run in test mode")("test131-fft", "run in test mode")("test60-sd", "run in test mode")("sect163r1", "attack signatures over sect163r1")("sect163r1-fft", "attack signatures over sect163r1")("prime192v1", "attack signatures over prime192v1")("prime192v1-fft", "attack signatures over prime192v1")("mtest", "memory test")("mtest-fft", "memory test for FFT")("strinput", "input file contains signatures in plain string format")("red", "perform reduction")("sd", "use sort-and-difference")("fft", "perform key recovery")("fft-mpi", "perform key recovery with distributed memory FFT")("fft-outlim", po::value<uint64_t>(), "how many top candidates to be written out")("fft-mpi-batch", po::value<uint32_t>(), "load input samples by batches")("fft-mpi-batch-uint32_t", po::value<uint32_t>(), "load input samples by batches")("known", po::value<int>(), "specify known MSBs of the secret key")("lim", po::value<uint32_t>(), "use only the first lim signatures as input")("gamma", po::value<int>(), "gamma")("mybound-bit", po::value<uint32_t>(), "force h values to be certain bound.")("in", po::value<std::string>(), "load signature data from a file")("out", po::value<std::string>(), "save reduced signature to a file with specified prefix")("dir", po::value<std::string>(), "specify directory for the files to be saved")("out-index", "save index info in each node right after the reduction")("fcount", po::value<int>(), "number of input files")("a-vec", po::value<std::vector<uint32_t>>()->multitoken(), "set a vector")("v-vec", po::value<std::vector<uint32_t>>()->multitoken(), "set v vector")("n-vec", po::value<std::vector<uint32_t>>()->multitoken(), "set n vector")("m-vec", po::value<std::vector<double>>()->multitoken(), "set m vector");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help"))
	{
		std::cout << desc << "\n";
		return EXIT_SUCCESS;
	}

	int log_prec = 100;
	if (vm.count("verbose"))
	{
		log_prec *= 100;
	}

	/* Parameters */
	uint32_t n_bit; // key length
	mpz_class n;	// modulas
	uint32_t a;
	uint32_t filter;
	double b;
	uint32_t gamma;
	uint32_t lim;
	uint64_t fft_outlim;
	size_t iota;					   // round num
	uint32_t l;						   //
	Domain pp;						   // cslass of key kength and secret key
	mpz_class d;					   // seacret key
	std::vector<SignatureSimple> sigs; // vector of signatures
	bool testmode = false;			   // currently only for memory test
	bool strinput = false;
	bool amplifyfirst = false;				 // is precomputation
	bool four_list_sum = true;				 // is 4-list sum algorithm
	bool sort_diff = false;					 // sort-diff or 4-list
	bool only_fft_mpi = false;				 // is only FFT-MPI
	std::vector<uint32_t> b_info;			 //
	std::vector<uint32_t> l_info;			 //
	uint32_t ub, kb;						 // unknow bits, known bnits
	mpz_class d_hi, d_lo;					 //
	uint32_t mybound_bit;					 //
	uint32_t batchsize;						 //
	std::vector<uint32_t> threshold_bit_vec; // vector of n
	std::vector<uint32_t> ignore_bit_vec;	 // vector of v
	std::vector<uint32_t> a_vec;			 // vector of a
	std::vector<double> m_vec;				 // vector of m

	if (vm.count("known"))
	{
		kb = vm["known"].as<int>();
	}
	else
	{
		kb = 0;
	}
	if (vm.count("gamma"))
	{
		gamma = vm["gamma"].as<int>();
	}
	else
	{
		gamma = 0;
	}
	if (vm.count("strinput"))
	{
		strinput = true;
	}

	if (vm.count("a-vec"))
	{
		a_vec = vm["a-vec"].as<std::vector<uint32_t>>();
	}

	if (vm.count("v-vec"))
	{
		threshold_bit_vec = vm["v-vec"].as<std::vector<uint32_t>>();
	}

	if (vm.count("n-vec"))
	{
		ignore_bit_vec = vm["n-vec"].as<std::vector<uint32_t>>();
	}

	if (vm.count("m-vec"))
	{
		m_vec = vm["m-vec"].as<std::vector<double>>();
	}

	if (vm.count("lim"))
	{
		lim = vm["lim"].as<uint32_t>();
	}
	else if (m_vec.size() != 0)
	{
		if (vm.count("red"))
		{
			lim = (int)std::pow(2.f, m_vec[0]);
		}
		else
		{

			lim = (int)std::pow(2.f, m_vec[m_vec.size() - 1]);
		}
	}
	else
	{

		lim = UINT32_MAX;
	}

	/* Set parameters */
	if (vm.count("test90"))
	{
		/* Use signature data from input file */
		n_bit = 90;
		n = (mpz_class(1) << n_bit) - 33;
		a = 12;
		filter = four_list_sum == false ? 10 : 0;
		iota = 2;
		l = a + 2;
		b_info = {30, 30};
		l_info = {l, 15, 15};
		pp = mock::setup(n_bit, n);
		d = mpz_class("924408261565060156037890712");
	}
	else if (vm.count("test90-mpi"))
	{
		/* Use signature data from input file */
		n_bit = 90;
		n = (mpz_class(1) << n_bit) - 33;
		a = 13;
		filter = 65;
		iota = 2;
		l = a + 2;
		b_info = {0};
		l_info = {l, 15};
		pp = mock::setup(n_bit, n);
		d = mpz_class("924408261565060156037890712");
	}
	else if (vm.count("test60"))
	{
		/* Use signature data from input file */
		n_bit = 60;
		n = (mpz_class(1) << n_bit) - 1061862795;
		a = 14;
		filter = 0;
		iota = 2;
		l = a + 2;
		b = 3 * a - 8;
		// b_info = {10,12};
		b_info = {24, 24};
		l_info = {l, 10, 12};
		// a_vec = {8, 16};
		// threshold_bit_vec = {6, 5};
		// ignore_bit_vec = {12, 28};
		// iota = a_vec.size();
		// b = (double)(n_bit-filter-kb-l)/iota;
		pp = mock::setup(n_bit, n);
		d = mpz_class("302038621189435203");
	}
	else if (vm.count("test10"))
	{
		/* Use signature data from input file */
		n_bit = 10;
		n = (mpz_class(1) << n_bit) - 27;
		a = 10;
		filter = 0;
		iota = 2;
		l = a + 2;
		b = 3 * a - 8;
		log_prec = 10;
		// b_info = {10,12};
		b_info = {1, 1};
		l_info = {l, 10, 20};
		// iota = a_vec.size();
		// b = (double)(n_bit-filter-kb-l)/iota;
		pp = mock::setup(n_bit, n);
		d = mpz_class("829");
	}
	else if (vm.count("test60-sd"))
	{
		n_bit = 60;
		n = (mpz_class(1) << n_bit) - 1061862795;
		a = 19;
		filter = 0;
		iota = 2;
		l = a + 2;
		b = a - 1;
		pp = mock::setup(n_bit, n);
		d = mpz_class("302038621189435203");
	}
	else if (vm.count("test131"))
	{
		n_bit = 131;
		n = (mpz_class(1) << n_bit) - 681;
		/* parameters for 1-bit bias */
		a = 21;
		filter = 0;
		iota = 2;
		l = a + 2;
		b_info = {35, 66};
		l_info = {l, 28, 26};
		pp = mock::setup(n_bit, n);
		d = mpz_class("2690717704635825153895385645374096400717"); // 2
																   // d = mpz_class("1647779926160126558973906702628759962437"); // 1
																   // d = mpz_class("1361129467683753853853498429727072845483"); // 0
	}
	else if (vm.count("test131-fft"))
	{
		/* Use sect163r1 */
		n_bit = 131;
		n = (mpz_class(1) << n_bit) - 681;
		a = 27;
		filter = 99;
		iota = 0;
		l = a + 2;
		b_info = {0};
		l_info = {l, 29};
		pp = mock::setup(n_bit, n);
		d = mpz_class("1361129467683753853853498429727072845483");
	}
	else if (vm.count("sd"))
	{
		/* Use real qDSA */
		n_bit = 252;
		n = (mpz_class(1) << 252) + mpz_class("27742317777372353535851937790883648493");
		/* parameters for 3-bit bias */
		a = 28;
		filter = 0;
		iota = 8;
		l = a + 2;
		b = a + 2 - gamma;
		pp = mock::setup(n_bit, n);
		d = mpz_class("5220582922658643192668885191618908575833980181104027493552863441828733052420");
	}
	else if (vm.count("sect163r1"))
	{
		/* Use sect163r1 */
		n_bit = 162;
		n = (mpz_class(1) << 162) - mpz_class("865766333097319309760613");
		/* parameters for 1-bit bias */
#if 1 // parameter for 97.3% accuracy, 30 bits known
	  // a = 22;
	  // filter = 0;
	  // iota = 2;
	  // l = a+2;
	  // b_info = {46,54};
	  // l_info = {l,29,30};
		a = 23;
		filter = 0;
		iota = 2;
		l = a + 2;
		b_info = {62, 73};
		l_info = {l, 28, 26};
#endif
#if 0 // parameter for 97.3% accuracy
		a = 22;
		filter = 0;
		iota = 2;
		l = a+2;
		b_info = {59,69};
		l_info = {l,29,29};
#endif
#if 0 // parameters for full 1-bit bias, 31 bits are known
		a = 21;
		filter = 0;
		iota = 2;
		l = a+2;
		b_info = {42,57};
		l_info = {l,29,27};
#endif
#if 0 // parameters for full 1-bit bias
		a = 21;
		filter = 0;
		iota = 2;
		l = a+2;
		b_info = {55,72};
		l_info = {l,29,27};
#endif
		pp = mock::setup(n_bit, n);
		d = mpz_class("4336966521141612760869415195855092141770523415923");
	}
	else if (vm.count("sect163r1-fft"))
	{
		/* Use sect163r1 */
		n_bit = 162;
		n = (mpz_class(1) << 162) - mpz_class("865766333097319309760613");
		/* parameters for 1-bit bias */
#if 1 // parameter for 97.3% accuracy, 30 bits known
		a = 28;
		filter = 46 + 54; // 2^32 FFT
#endif
#if 0 // parameter for 97.3% accurary
		a = 27;
		filter = 128;
#endif
#if 0 // parameters for full 1-bit bias, 31 bits are known
		a = 25;
		filter = 42+57; // 2^32 FFT
#endif
#if 0 // parameters for full 1-bit bias
		a = 25;
		filter = 127; // 2^35 FFT
#endif
		iota = 0;
		l = a + 2;
		b_info = {0};
		l_info = {l, 29};
		pp = mock::setup(n_bit, n);
		d = mpz_class("4336966521141612760869415195855092141770523415923");
	}
	else if (vm.count("prime192v1"))
	{
		/* Use prime192v1 */
		n_bit = 192;
		n = (mpz_class(1) << 192) - mpz_class("31607402335160671281192228815");
		/* parameters for 1-bit bias */
#if 0 // 6-bit filtered, 99% error, 145 bits known
		a = 23;
		filter = 6;
		iota = 1;
		l = a+2;
		b = 9;
		sort_diff = true;
#endif
#if 0 // 6-bit filtered, 99% error, 117 bits known
		a = 23;
		filter = 6;
		iota = 1;
		l = a+2;
		b_info = {37};
		l_info = {l,25};
#endif
#if 0 // 6-bit filtered, 99% error, 89 bits known
		a = 27;
		filter = 6;
		iota = 1;
		l = a+2;
		b_info = {65};
		l_info = {l,22};
#endif
#if 0 // 6-bit filtered, 99% error, 61 bits known
		a = 25;
		filter = 6;
		iota = 2;
		l = a+2;
		b_info = {46,47};
		l_info = {l,28,29};
#endif
#if 0 // 6-bit filtered, 99% error, 33 bits known
		a = 27;
		filter = 6;
		iota = 2;
		l = a+2;
		b_info = {61,60};
		l_info = {l,29,29};
#endif
#if 0 // 6-bit filtered, 99% error, 0 bits known
		a = 27;
		filter = 6;
		iota = 2;
		l = a+2;
		b_info = {75,74};
		l_info = {l,29,30};
#endif
#if 0 // no filter, no error
		a = 27;
		filter = 0;
		iota = 2;
		l = a+2;
		b_info = {76,76};
		l_info = {l,29,29};
#endif
		pp = mock::setup(n_bit, n);
		d = mpz_class("3596558503072788033269653289719865785571921581759751324361");
	}
	else if (vm.count("prime192v1-fft"))
	{
		/* Use prime192v1 */
		n_bit = 192;
		n = (mpz_class(1) << 192) - mpz_class("31607402335160671281192228815");
		/* parameters for 1-bit bias */
#if 0 // 6-bit filtered, 99% error, 145 bits known, 2^32 FFT
		a = 23;
		filter = 15;
#endif
#if 0 // 6-bit filtered, 99% error, 117 bits known, 2^32 FFT
		a = 23;
		filter = 37+6;
#endif
#if 0 // 6-bit filtered, 99% error, 89 bits known, 2^32 FFT
		a = 20;
		filter = 65+6;
#endif
#if 0 // 6-bit filtered, 99% error, 61 bits known, 2^32 FFT
		a = 23;
		filter = 46+47+6;
#endif
#if 0 // 6-bit filtered, 99% error, 33 bits known, 2^32 FFT
		a = 27;
		filter = 61+60+6;
#endif
#if 0 // 6-bit filtered, 99% error, 0 bits known, 2^37 FFT
		a = 28;
		filter = 75+74+6;
#endif
#if 1 // no filter, no error, 0 bits known, 2^38 FFT
		a = 25;
		filter = 76 + 76 + 2;
#endif
		iota = 0;
		l = a + 2;
		b_info = {0};
		l_info = {l, 30};
		pp = mock::setup(n_bit, n);
		d = mpz_class("3596558503072788033269653289719865785571921581759751324361");
	}
	else
	{
		/* Use real qDSA */
		n_bit = 252;
		n = (mpz_class(1) << 252) + mpz_class("27742317777372353535851937790883648493");
		/* parameters for 3-bit bias */
		/*
		 a = 21;
		 filter = 0;
		 iota = 4;
		 ofst_info = {2,2,1,0};
		 */
		/* parameters for 2-bit bias */
		/*
		 a = 24;
		 filter = 19;
		 iota = 3;
		 ofst_info = {2,1,0};

		 l = a+2;
		 b = (double)(n_bit-filter-kb-l)/iota;
		 */
		/* parameters for reduction experiments: test252_a15_b0_f0 */
		a = 15;
		filter = 0;
		iota = 5;
		l = a + 2;
		b = 3 * a - 1.59;
		pp = mock::setup(n_bit, n);
		d = mpz_class("5220582922658643192668885191618908575833980181104027493552863441828733052420");
	}
	if (myrank == master)
	{
		gmp_printf("pp=(n_bit=%u, n=%Zd),\n        d=%Zd\n", pp.n_bit, pp.n.get_mpz_t(), d.get_mpz_t());
	}

	uint64_t num_sigs = m_vec.size() == 0 ? (1U << l) : lim;

	/* Decide batch size for fft-mpi */
	if (vm.count("fft-mpi-batch"))
	{
		batchsize = (1UL << vm["fft-mpi-batch"].as<uint32_t>());
	}
	else if (vm.count("fft-mpi-batch-uint32_t"))
	{
		batchsize = (vm["fft-mpi-batch-uint32_t"].as<uint32_t>());
	}
	else
	{
		batchsize = num_sigs;
	}

	/* Load signature data */
	mpz_class bound = mpz_class(1) << (n_bit - filter);
	std::string fname;
	ub = n_bit - kb;
	d_hi = (d >> ub) << ub;
	d_lo = d - d_hi;

	if (four_list_sum == true)
	{
		l = (a_vec[0] + 2); // 4-list sum algorithmのときのみ
	}

	// for (int i = 0;  i < (1<<5); i++) { // for each 20MSB candidate of sk
	if (vm.count("in") && vm.count("fft-mpi") == 0)
	{
		sigs.reserve(num_sigs);
		if (myrank == master || vm.count("fft-mpi"))
		{
			int fcount = 1;
			if (vm.count("fcount"))
			{
				fcount = vm["fcount"].as<int>();
				for (int fc = 0; fc < fcount; fc++)
				{
					fname = vm["in"].as<std::string>() + "_" + std::to_string(fc) + ".bin";
					printf("[%d]/[%d]: loading %s\n", myrank, worldsize, fname.c_str());
					sigload(sigs, fname, num_sigs, strinput, bound);
				}
			}
			else
			{
				fname = vm["in"].as<std::string>() + ".bin";
				printf("[%d]/[%d]: loading %s\n", myrank, worldsize, fname.c_str());
				sigload(sigs, fname, num_sigs, strinput, bound);
			}
			/* Preprocess sigs for recovering remaining bits */
			if (kb != 0 && vm.count("red"))
			{
				printf("[%d]/[%d]: assuming %u bits of sk are known; rewriting samples.\n", myrank, worldsize, kb);
				for (SignatureSimple &sig : sigs)
				{
					mpz_class tmp = sig.s + sig.h * d_hi;
					mpz_mod(sig.s.get_mpz_t(), tmp.get_mpz_t(), n.get_mpz_t());
				}
			}
			/* For mtest-fft: force upper bound of h to mybound */
			if (vm.count("mybound-bit"))
			{
				mybound_bit = vm["mybound-bit"].as<uint32_t>();
				for (SignatureSimple &sig : sigs)
				{
					sig.h = sig.h >> (n_bit - filter - mybound_bit);
				}
			}

			printf("[%d]/[%d]: loaded %lu signatures\n", myrank, worldsize, sigs.size());
#if 0
			//std::sort(sigs.begin(), sigs.end());
			sigvprint(sigs, 0, 10);
#endif
#if 0
			countbias(sigs, pp, d, mpz_class(0), mpz_class(1)<<(n_bit-1)); // check if k is really 1-bit biased.
#endif
		}
	}
	else if (vm.count("in") && vm.count("fft-mpi"))
	{
		if (vm.count("fft-mpi-batch-uint32_t"))
		{
			batchsize = (vm["fft-mpi-batch-uint32_t"].as<uint32_t>());
		}
		else
		{
			int mp = 3 * a_vec.back() + threshold_bit_vec.back() - ignore_bit_vec.back() - 2;
			batchsize = int((1ULL << mp) / worldsize);
		}
		// batchsize = (1ULL << (mp+1));
		sigs.reserve(batchsize);
		fname = vm["in"].as<std::string>() + ".bin";
		printf("[%d]/[%d]: will load signatures by batch of size %u\n", myrank, worldsize, batchsize);
	}
	else
	{
		return EXIT_SUCCESS;
	}
	std::string out_prefix, dir;
	if (vm.count("out"))
		out_prefix = vm["out"].as<std::string>();
	bool out_index = false;
	if (vm.count("out-index"))
		out_index = true;
	if (vm.count("dir"))
		dir = vm["dir"].as<std::string>();

	// output parameters
	if (myrank == master)
	{
		for (int i = 0; i < a_vec.size(); i++)
		{
			std::cout << "a[" << i << "] = " << a_vec[i] << std::endl;
		}

		for (int i = 0; i < threshold_bit_vec.size(); i++)
		{
			std::cout << "v[" << i << "] = " << threshold_bit_vec[i] << std::endl;
		}

		for (int i = 0; i < ignore_bit_vec.size(); i++)
		{
			std::cout << "n[" << i << "] = " << ignore_bit_vec[i] << std::endl;
		}
		for (int i = 0; i < m_vec.size(); i++)
		{
			std::cout << "m[" << i << "] = " << m_vec[i] << std::endl;
		}
	}

	/* REDUCTION */
	if (vm.count("red"))
	{
		// std::cout << "reduction" << std::endl;
		// std::cout << "amplifyfirst: " << amplifyfirst << std::endl;
		if (four_list_sum == true)
		{
			// std::cout << "myrank == master: " << (myrank == master) << std::endl;
			if (myrank == master)
			{
				std::cout << "bef four list sum" << std::endl;
				// exhaustive_four_sum(sigs, n_bit - filter - b_info[0], 1UL << l_info[1], 100);
				std::cout << "size is " << sigs.size() << std::endl;
				iterative_HGJ_four_list_sum(sigs, threshold_bit_vec, ignore_bit_vec, a_vec, m_vec, pp.n_bit, 100, out_prefix, dir);
			}
			l = l_info[1];
			filter += b_info[0];
			a = l - 2;
			l_info.erase(l_info.begin(), l_info.begin() + 1);
			b_info.erase(b_info.begin(), b_info.begin() + 1);
			iota--;
			bound = mpz_class(1) << (n_bit - filter);
		}
		// world.barrier();
		if (four_list_sum == false)
		{
			if (sort_diff)
			{
				sort_and_difference(sigs, bound, n_bit, l, b, filter, a, log_prec, iota, out_prefix, dir, testmode);
			}
			else
			{
				schroeppel_shamir_mpi(sigs, bound, n_bit, l, b, filter, a, log_prec, b_info, l_info, iota, out_prefix, out_index, dir, testmode);
			}
		}
	}
	world.barrier();
	std::cout << "end reduction" << std::endl;
	if (!vm.count("fft-mpi") && !vm.count("fft"))
	{
		std::cout << "do nothing for recovery key." << std::endl;
		return EXIT_SUCCESS;
	}
	/* FFT */
	// bound for h/2^kb
	bound = bound >> kb;
	uint64_t C = bound.get_ui(); // FFTのテーンブルサイズ
	std::cout << "start sort sigs" << std::endl;
	std::sort(sigs.begin(), sigs.end());
	std::cout << "end sort sigs" << std::endl;
	if (vm.count("mybound-bit"))
	{
		C = (1ULL << mybound_bit);
	}
	if (four_list_sum == true)
	{
		C = 1ULL;
		C = C << (pp.n_bit - std::accumulate(ignore_bit_vec.begin(), ignore_bit_vec.end(), 0));
		std::cout << "C: " << C << std::endl;
	}
	if (vm.count("fft-outlim"))
	{
		fft_outlim = vm["fft-outlim"].as<uint64_t>();
	}
	else
	{
		fft_outlim = C;
	}
	if (vm.count("fft-mpi"))
	{

		const uint32_t L = batchsize;

		std::cout << "L size: " << L << std::endl;
		compute_bias_mpi(C, L, kb, ub, d, d_hi, pp, sigs, fft_outlim, batchsize, fname, dir);
	}
	else if (vm.count("fft") && myrank == master)
	{
		if (!vm.count("red"))
		{ // reductionをしなかった場合
			fname = vm["in"].as<std::string>() + ".bin";
			printf("[%d]/[%d]: loading %s\n", myrank, worldsize, fname.c_str());
			sigload(sigs, fname, num_sigs, strinput, bound);
		}

		/* Debug Logging */
#if 0
		std::sort(sigs.begin(), sigs.end());
		sigvprint(sigs, 0, 10);
#endif
		/* FFT */
		const uint32_t L = sigs.size();
		// bound for h/2^kb
		if (four_list_sum == true)
		{
			C = 1ULL;
			C = C << (pp.n_bit - std::accumulate(ignore_bit_vec.begin(), ignore_bit_vec.end(), 0));
			std::cout << "C: " << C << std::endl;
		}
		printf("Initializing FFT table of size %lu...\n", C);
		fftw_complex *W = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * C); // FFTテーブル
		if (W == NULL)
		{
			printf("malloc failed\n");
			exit(1);
		}
		std::cout << "sigs.size(): " << sigs.size() << std::endl;
		compute_bias(W, C, L, kb, pp, sigs);

		/* Find peak and compute noise */
		double peak;
		uint64_t peak_at;
		std::priority_queue<WCandidate,
							std::vector<WCandidate>,
							std::greater<WCandidate>>
			heap;
		printf("Looking for the peak...\n");
		find_peak(W, C, L, fft_outlim, peak, peak_at, heap);
		printf("Computing average noise...\n");
		double noise = compute_noise_avg(W, C, L, peak_at);

		/* Compute W_m = m*n/C */
		mpz_class w;
		if (kb == 0)
			w = (mpz_class)peak_at * n / C;
		else
			w = (mpz_class)peak_at * (mpz_class(1) << ub) / C + d_hi;
		mpf_class w_f = mpf_class(w);
		mpf_class d_f = mpf_class(d);
		mpf_class rel_error = abs(w_f - d_f) / d_f;
		std::cout << "peak_at:" << peak_at << ", peak: " << peak << ", n: " << n << ", C: " << C << std::endl;
		gmp_printf("Estimated secret   w = %Zd \n"
				   "Real secret        d = %Zd \n"
				   "Relative error       = %.Ff \n",
				   w.get_mpz_t(), d.get_mpz_t(), rel_error);

		printf("Average noise             = %lf\n"
			   "Estimated noise 1/sqrt(L) = %lf\n",
			   noise, 1 / sqrt(L));

		// Comparison with the actual secret
		std::string dbin = d.get_str(2);
		std::string wbin = w.get_str(2);
		int count_msb = 0;
		int idx = 0;
		while (1)
		{
			if (dbin[idx] != wbin[idx] || idx >= n_bit)
				break;
			count_msb++;
			idx++;
		}
		std::cout << dbin << std::endl;
		std::cout << wbin << std::endl;
		printf("Recovered %d-MSBs of sk d\n", count_msb);

#ifdef SAVE_BIAS
		/* Saving bias to file */
		std::string bias_fname("bias.csv");
		printf("saving top %lu bias candidates to file %s\n", fft_outlim, bias_fname.c_str());
		std::ofstream bias_file(bias_fname.c_str(), std::ofstream::trunc);
		while (!heap.empty())
		{
			WCandidate w = heap.top();
			bias_file << w.pos << "," << w.norm << std::endl;
			heap.pop();
		}
		bias_file.close();
#endif
		printf("Freeing up FFT table...\n");
		fftw_free(W);

		/* Compute FFT of h */
#ifdef FFT_h
		printf("Computing bias of h..\n");
		fftw_complex *Wh = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * C);
		compute_bias_h(Wh, C, L, kb, pp, sigs);
		printf("Wh0 = %lf \n", compute_norm(Wh[0], L));

		/* Find peak */
		double h_peak;
		uint64_t h_peak_at;
		std::priority_queue<WCandidate,
							std::vector<WCandidate>,
							std::greater<WCandidate>>
			heap_h;
		find_peak(Wh, C, L, fft_outlim, h_peak, h_peak_at, heap_h);
		double ffth_max = compute_norm(Wh[h_peak_at], L);
		printf("FFT peak found ffth_max = %lf \n", ffth_max);
#endif

		/* Saving bias to file */
#ifdef SAVE_BIAS
		std::string ffth_fname("fft_h.csv");
		printf("saving top %lu fft_h to file %s\n", fft_outlim, ffth_fname.c_str());
		std::ofstream ffth_file(ffth_fname.c_str(), std::ofstream::trunc);
		while (!heap_h.empty())
		{
			WCandidate w = heap_h.top();
			ffth_file << w.pos << "," << w.norm << std::endl;
			heap_h.pop();
		}
		ffth_file.close();
#endif
#ifdef FFT_h
		printf("Freeing up FFT table...\n");
		fftw_free(Wh);
#endif
	}
	//} // end for
	if (myrank == master)
		printf("attack_mpi is done\n");

	return EXIT_SUCCESS;
}