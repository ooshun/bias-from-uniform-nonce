#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdint.h>
#include <omp.h>
#include <malloc.h>
#include <iostream>
#include <algorithm>
#include <queue>
#include <bitset>
#include <chrono>
#include <numeric>

#include <gmpxx.h>
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include "mocksig.h"
#include "reduction.h"

// #define debug_four_list
// #define use_break_flag
#define ALL_SIGNATURES
// #define ALL_lists

namespace mpi = boost::mpi;
namespace spsort = boost::sort::spreadsort;

LRComb::LRComb(mpz_class hh, uint32_t i, uint32_t j)
{
	hsum = hh;
	idx_L = i;
	idx_R = j;
}

/* serialize user-defined types */
namespace boost
{
	namespace serialization
	{
		template <class Archive>
		void serialize(Archive &ar, gs &scalar, const unsigned int version)
		{
			ar & scalar.v;
		}

		template <class Archive>
		void serialize(Archive &ar, gss &scalar, const unsigned int version)
		{
			ar & scalar.v;
		}
	}
}

void idxsave(std::vector<Index> &is, const std::string &filename)
{
	FILE *fp = fopen(filename.c_str(), "wb");

	for (auto &i : is)
	{
		fwrite(&i.idx_L1, sizeof(uint32_t), 1, fp);
		fwrite(&i.idx_R1, sizeof(uint32_t), 1, fp);
		fwrite(&i.idx_L2, sizeof(uint32_t), 1, fp);
		fwrite(&i.idx_R2, sizeof(uint32_t), 1, fp);
		fwrite(&i.flip, sizeof(bool), 1, fp);
	}
	fclose(fp);
}

void idxload(std::vector<Index> &is, const std::string &filename)
{
	FILE *fp = fopen(filename.c_str(), "rb");
	Index i;
	while (fread(&i.idx_L1, sizeof(uint32_t), 1, fp) && fread(&i.idx_R1, sizeof(uint32_t), 1, fp) && fread(&i.idx_L2, sizeof(uint32_t), 1, fp) && fread(&i.idx_R2, sizeof(uint32_t), 1, fp) && fread(&i.flip, sizeof(bool), 1, fp))
	{
		is.emplace_back(i);
	}
	fclose(fp);
}

std::string msb(mpz_class m, uint32_t a, uint32_t bit)
{
	std::string bin = m.get_str(2);
	uint32_t padding = bit - bin.length();
	if (padding != 0)
		bin = std::string(padding, '0') + bin;

	return bin.substr(0, a);
}

/* helper method that collects linear combinations of two whose a-MSB is equal to A */
void collect_lc_two(std::vector<LRComb64> &combs, const std::vector<uint128_t> &L, const std::vector<uint128_t> &R,
					const uint32_t &A, const uint32_t &a, const uint32_t &current_threshold_bit, const int &ofst, const size_t &pad,
					const uint32_t &lim)
{
	uint32_t i = 0, j = 0;
	bool flag_i = false;
	bool flag_j = false;
	uint32_t bad = 0;
	size_t lshift = 128 - (a + 1) - pad;
	size_t rshift = 128 - (a + 1) - 64;
	uint128_t one_128 = 1;
	uint128_t A_128 = (uint128_t)A;
	uint128_t A0 = A_128 << lshift;
	uint128_t A1_low = (one_128 << lshift) - 1;
	uint128_t A1 = A0 + A1_low;
	uint128_t Amid = A0 + A1_low / 2;
#if 0
	print_uint128(A_128);
	print_uint128(A0);
	print_uint128(Amid);
	print_uint128(A1);
#endif

	uint32_t lsize = L.size();
	uint32_t rsize = R.size();

	uint128_t sum = L[i] + R[j];
	while (combs.size() < lim)
	{
		if (sum < A0)
		{
			if (j == rsize - 1)
				break;
			j++;
		}
		else if (A1 < sum)
		{
			if (i == lsize - 1)
				break;
			i++;
		}
		else
		{
			combs.emplace_back((uint64_t)((sum << pad) >> rshift), i, j);
			/* check if indices are at the end */
			if (i == lsize - 1 && j == rsize - 1)
				break;
			else if (i == lsize - 1)
			{
				j++;
				sum = L[i] + R[j];
				continue;
			}
			else if (j == rsize - 1)
			{
				i++;
				sum = L[i] + R[j];
				continue;
			}

			uint128_t peek_i, peek_j;
			peek_i = L[i + 1] + R[j];
			peek_j = L[i] + R[j + 1];
			flag_i = (A0 <= peek_i) && (peek_i <= A1);
			flag_j = (A0 <= peek_j) && (peek_j <= A1);

			if (flag_i ^ flag_j)
			{
				if (flag_i)
					i++;
				else
					j++;
			}
			else
			{
				if (flag_i && flag_j)
					bad++;
				uint32_t c = 1;
				if (sum < Amid)
				{
					while (flag_i && i + c < lsize && combs.size() < lim)
					{
						peek_i = L[i + c] + R[j];
						flag_i = (A0 <= peek_i) && (peek_i <= A1);
						if (flag_i)
							combs.emplace_back((uint64_t)((peek_i << pad) >> rshift), i + c, j);
						c++;
					}
					j++;
				}
				else
				{
					while (flag_j && j + c < rsize && combs.size() < lim)
					{
						peek_j = L[i] + R[j + c];
						flag_j = (A0 <= peek_j) && (peek_j <= A1);
						if (flag_j)
							combs.emplace_back((uint64_t)((peek_j << pad) >> rshift), i, j + c);
						c++;
					}
					i++;
				}
			}
		} // else
		sum = L[i] + R[j];
	} // while (combs.size() < lim)
}

void exhaustive_four_sum(std::vector<SignatureSimple> &sigs, const uint32_t threshold_bit, const uint32_t keep_max, const int log_prec)
{
	// std::cout << "four list sum algorithms beginning" << std::endl;
	mpi::environment env;
	mpi::communicator world;
	const int master = 0;
	const int myrank = world.rank();
	const int worldsize = world.size();

	uint32_t S, q1, q2, q3;
	mpz_class threshold_mpz = mpz_class(1) << threshold_bit;
	std::vector<SignatureSimple> result;
	result.reserve(keep_max);
	uint32_t local_keep_max = keep_max / omp_get_max_threads();
	uint32_t n_threads = omp_get_max_threads();

	/* Split sigs into L1 || R1 || L2 || R2 */
	S = sigs.size();
	q1 = S / 4;
	q2 = S / 2;
	q3 = S * 3 / 4;
	gmp_printf("[%d]/[%d]: exhaustively looking for %u sums with threshold = %Zd\n",
			   myrank, worldsize, keep_max, threshold_mpz.get_mpz_t());
	printf("[%d]/[%d]: using %u threads\n", myrank, worldsize, n_threads);
#pragma omp parallel shared(sigs, result)
	{
		bool break_flag = false;
		mpz_class h, s, hi, hj, hk, hl, si, sj, sk, sl;
		std::vector<SignatureSimple> local_result;
		local_result.reserve(local_keep_max);

#pragma omp for schedule(static)
		for (uint32_t i = 0; i < q1; i++)
		{
			if (break_flag)
				continue;
			hi = sigs[i].h;
			si = sigs[i].s;
			for (uint32_t j = q1; j < q2; j++)
			{
				if (break_flag)
					continue;
				hj = sigs[j].h;
				sj = sigs[j].s;
				for (uint32_t k = q2; k < q3; k++)
				{
					if (break_flag)
						continue;
					hk = sigs[k].h;
					sk = sigs[k].s;
					for (uint32_t l = q3; l < S; l++)
					{
						if (break_flag)
							continue;
						hl = sigs[l].h;
						sl = sigs[l].s;
						h = hi + hj - hk - hl;
						if (h >= 0 && h < threshold_mpz)
						{
							local_result.emplace_back(SignatureSimple(h, si + sj - sk - sl));
						}
						else if (h < 0 && h > -threshold_mpz)
						{
							local_result.emplace_back(SignatureSimple(-h, sk + sl - si - sj));
						}
						if (local_result.size() >= local_keep_max)
						{
							break_flag = true;
						}
					}
				}
			}
		}
#pragma omp critical
		{
			result.insert(result.end(), local_result.begin(), local_result.end());
		}
	} // pragma omp parallel

	printf("[%d]/[%d]: done!\n", myrank, worldsize);
	printf("[%d]/[%d]: freeing up memory..\n", myrank, worldsize);
	std::vector<SignatureSimple>().swap(sigs);
	malloc_trim(0);
	if (result.size() < keep_max)
	{
		printf("[%d]/[%d]: WARNING: found only %lu < %u collisions \n", myrank, worldsize, result.size(), keep_max);
	}
	else
	{
		printf("[%d]/[%d]: found %lu collisions\n", myrank, worldsize, result.size());
	}
	printf("[%d]/[%d]: copying result..\n", myrank, worldsize);
	sigs.reserve(result.size());
	std::copy(result.begin(), result.end(), std::back_inserter(sigs));
}

void iterative_HGJ_four_list_sum(std::vector<SignatureSimple> &sigs, const std::vector<uint32_t> threshold_bit_vec,
								 const std::vector<uint32_t> ignore_bit_vec, const std::vector<uint32_t> a_vec, const std::vector<double> m_vec, const uint32_t l_bit,
								 const uint32_t keep_max, std::string out_prefix = "", std::string dir = "")
{
	/* Start measuring time */
	printf("reduction started\n");
	auto start = std::chrono::high_resolution_clock::now();

#ifdef ALL_lists
	std::vector<SignatureSimple> sigs1, sigs2;

	sigs1.insert(sigs1.end(), sigs.begin(), sigs.begin() + sigs.size() / 2);
	sigs2.insert(sigs2.end(), sigs.begin() + sigs.size() / 2, sigs.end());
	sigs.clear();
	sigs.shrink_to_fit();
	std::cout << "sigs1.size(): " << sigs1.size() << std::endl;
	std::cout << "sigs2.size(): " << sigs2.size() << std::endl;

	int i = 0;
	auto start_round = std::chrono::high_resolution_clock::now();
	parametarized_four_list_sum(sigs1, threshold_bit_vec[i], ignore_bit_vec[i], a_vec[i], m_vec[i], m_vec[i + 1], l_bit - std::accumulate(ignore_bit_vec.begin(), ignore_bit_vec.begin() + i, 0), 100, (i + 1), dir, out_prefix);
	auto end_round = std::chrono::high_resolution_clock::now();
	printf("-------------------------------------------------- Round %lu finished --------------------------------------------------\n", (i + 1));
	std::chrono::duration<double> elapsed_round = end_round - start_round;
	printf("Elapsed time: %1.f seconds at round %lu \n", elapsed_round.count(), (i + 1));

	start_round = std::chrono::high_resolution_clock::now();
	parametarized_four_list_sum(sigs2, threshold_bit_vec[i], ignore_bit_vec[i], a_vec[i], m_vec[i], m_vec[i + 1], l_bit - std::accumulate(ignore_bit_vec.begin(), ignore_bit_vec.begin() + i, 0), 100, (i + 1), dir, out_prefix);
	end_round = std::chrono::high_resolution_clock::now();
	printf("-------------------------------------------------- Round %lu finished --------------------------------------------------\n", (i + 1));
	elapsed_round = end_round - start_round;
	printf("Elapsed time: %1.f seconds at round %lu \n", elapsed_round.count(), (i + 1));

	sigs.insert(sigs.end(), sigs1.begin(), sigs1.end());
	sigs.insert(sigs.end(), sigs2.begin(), sigs2.end());
	sigs1.clear();
	sigs1.shrink_to_fit();
	sigs2.clear();
	sigs2.shrink_to_fit();
	std::cout << "sigs.size(): " << sigs.size() << std::endl;

	i = 1;
	start_round = std::chrono::high_resolution_clock::now();
	parametarized_four_list_sum(sigs, threshold_bit_vec[i], ignore_bit_vec[i], a_vec[i], m_vec[i], m_vec[i + 1], l_bit - std::accumulate(ignore_bit_vec.begin(), ignore_bit_vec.begin() + i, 0), 100, (i + 1), dir, out_prefix);
	end_round = std::chrono::high_resolution_clock::now();
	printf("-------------------------------------------------- Round %lu finished --------------------------------------------------\n", (i + 1));
	elapsed_round = end_round - start_round;
	printf("Elapsed time: %1.f seconds at round %lu \n", elapsed_round.count(), (i + 1));
#endif

	for (int i = 0; i < ignore_bit_vec.size(); i++)
	{
		std::cout << "4 list sum algorithm's round: " << (i + 1) << std::endl;
		printf("-------------------------------------------------- Round %lu begins --------------------------------------------------\n", (i + 1));
		std::cout << "l_bit - std::accumulate(ignore_bit_vec.begin(), ignore_bit_vec.begin() + i, 0): " << (l_bit - std::accumulate(ignore_bit_vec.begin(), ignore_bit_vec.begin() + i, 0)) << std::endl;

		auto start_round = std::chrono::high_resolution_clock::now();
		parametarized_four_list_sum(sigs, threshold_bit_vec[i], ignore_bit_vec[i], a_vec[i], m_vec[i], m_vec[i + 1], l_bit - std::accumulate(ignore_bit_vec.begin(), ignore_bit_vec.begin() + i, 0), 100, (i + 1), dir, out_prefix);
		auto end_round = std::chrono::high_resolution_clock::now();
		printf("-------------------------------------------------- Round %lu finished --------------------------------------------------\n", (i + 1));
		std::chrono::duration<double> elapsed_round = end_round - start_round;
		printf("Elapsed time: %1.f seconds at round %lu \n", elapsed_round.count(), (i + 1));
	}

	/* End measuring time */
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	printf("Elapsed time: %1.f seconds\n", elapsed.count());

	std::cout << "iterative_HGJ_four_list_sum is ended" << std::endl;
	std::cout << "sigs.size(): " << sigs.size() << std::endl;

	// if (dir.length()) {
	// 	dir = dir + "/";
	// }

	// if (out_prefix.length())
	// {
	// 	// file format: redsigs_sd_round-i.bin
	// 	std::string outsig = out_prefix + "_reduction" + ".bin";
	// 	printf("saving signatures to %s... \n", (dir + outsig).c_str());
	// 	sigsave(sigs, dir + outsig);
	// }
}

void parametarized_four_list_sum(std::vector<SignatureSimple> &sigs, const uint32_t threshold_bit, const uint32_t ignore_bit, const uint32_t a, const double m_double, const double mp, const uint32_t l_bit, const uint32_t keep_max, const int round, std::string dir = "", std::string out_prefix = "")
{
	mpi::environment env;
	mpi::communicator world;
	const int master = 0;
	const int myrank = world.rank();
	const int worldsize = world.size();

	uint32_t S, q1, q2, q3, a_bit;
	const std::string fname_prefix = "four_list_mpi";
	if (dir.length())
	{
		dir = dir + "/";
	}

	S = std::floor(std::pow(2, m_double));
	if (S > sigs.size())
	{
		std::cout << "signatures are smaller than S" << std::endl;
		S = sigs.size();
	}
#ifdef ALL_SIGNATURES
	S = sigs.size();
	std::cout << "Use all signatures." << std::endl;
#endif

	// resize size of sigs
	sigs.resize(S);

	int ofst;
	size_t pad;
	// File format: ss-mpi_round-i_{L1,L2,L3,L4}.bin
	std::string L1_fname, L2_fname, L3_fname, L4_fname;
	L1_fname = fname_prefix + "_round-" + std::to_string(round) + "_L1.bin";
	L2_fname = fname_prefix + "_round-" + std::to_string(round) + "_L2.bin";
	L3_fname = fname_prefix + "_round-" + std::to_string(round) + "_L3.bin";
	L4_fname = fname_prefix + "_round-" + std::to_string(round) + "_L4.bin";
	std::cout << "L1_fname :" << L1_fname << std::endl;
	std::vector<uint128_t> L1_h, L2_h, L3_h, L4_h;
	std::vector<uint64_t> L1_hhigh, L2_hhigh, L3_hhigh, L4_hhigh, L1_hlow, L2_hlow, L3_hlow, L4_hlow;
	uint32_t lower_threshold_uint, upper_threshold_uint, _keep_max;
	uint64_t sub_Lxp_max, threshold_uint, Lp_max, upper_sigs;
	uint128_t a_bit_max_128;
	mpz_class a_bit_max_mpz, threshold_mpz;
	std::vector<Index> index_result;
	double sig_percent = 0.01;

	uint32_t n_threads = omp_get_max_threads();
	printf("[%d]/[%d]: using %u threads\n", myrank, worldsize, n_threads);

	if (myrank == master)
	{
		q1 = S / 4;
		q2 = S / 2;
		q3 = S * 3 / 4;
		a_bit = std::min((int)std::floor(std::log2(q1)), (int)a);
		std::cout << "std::floor(std::log2(q1)): " << std::floor(std::log2(q1)) << ", a: " << a << ", a_bit: " << a_bit << std::endl;
		std::cout << "threshold_bit: " << threshold_bit << std::endl;
		uint32_t threshold_bit_tmp = std::min(threshold_bit, a_bit + 1);
		a_bit_max_mpz = (mpz_class(1) << (a_bit - 1)) - 1;
		a_bit_max_128 = (1LL << a_bit) - 1;
		threshold_mpz = mpz_class(1) << threshold_bit_tmp;
		_keep_max = std::pow(2, 3 * a_bit + threshold_bit_tmp - ignore_bit - 2); // 改良箇所 -2
		// uint64_t _keep_max = calc_num_out(a_bit, threshold_bit_tmp, ignore_bit);
		std::cout << "_keep_max:" << _keep_max << std::endl;
		uint32_t local_keep_max = keep_max / omp_get_max_threads();
		upper_sigs = std::pow(2, mp);
		std::cout << "upper_sigs: " << upper_sigs << std::endl;
		sub_Lxp_max = std::pow(2, a_bit + 1);
		threshold_uint = 1 << threshold_bit_tmp;
		// uint32_t threshold_uint = 1 << (threshold_bit + 1);
		std::cout << "threshold_uint:" << threshold_uint << std::endl;
		Lp_max = std::pow(2, 3 * a_bit - ignore_bit); // 改良箇所 -2
		std::cout << "Lp_max: " << Lp_max << std::endl;
		gmp_printf("[%d]/[%d]: looking for %u sums with threshold = %Zd\n",
				   myrank, worldsize, _keep_max, threshold_mpz.get_mpz_t());
		printf("[%d]/[%d]: using %u threads\n", myrank, worldsize, n_threads);

		compute_ofst_uint128(ofst, pad, l_bit, 1);
		std::cout << "ofst: " << ofst << ", pad: " << pad << std::endl;
	}
	// world.barrier();

	if (myrank == master)
	{
		/** sort */
		std::cout << "start sort L1." << std::endl;
		std::sort(sigs.begin(), sigs.begin() + q1);
		std::cout << "finished sort L1." << std::endl;
		std::cout << "start sort L2." << std::endl;
		std::sort(sigs.begin() + q1, sigs.begin() + q2);
		std::cout << "finished sort L2." << std::endl;
		std::cout << "start sort L3." << std::endl;
		std::sort(sigs.begin() + q2, sigs.begin() + q3);
		std::cout << "finished sort L3." << std::endl;
		std::cout << "start sort L4." << std::endl;
		std::sort(sigs.begin() + q3, sigs.begin() + S);
		std::cout << "finish sort L4" << std::endl;

		std::cout << "num of S: " << S << ", sigs.size(): " << sigs.size() << std::endl;
#ifdef debug_four_list
		for (int i = 0; i < q1; i++)
		{
			std::cout << "List1: i: " << i
					  << ", hi = " << sigs[i].h << ", si = " << sigs[i].s << std::endl;
		}
		for (int i = q1; i < q2; i++)
		{
			std::cout << "List2: i: " << (i - q1)
					  << ", hi = " << sigs[i].h << ", si = " << sigs[i].s << std::endl;
		}
		for (int i = q2; i < q3; i++)
		{
			std::cout << "List3: i: " << (i - q2)
					  << ", hi = " << sigs[i].h << ", si = " << sigs[i].s << std::endl;
		}
		for (int i = q3; i < S; i++)
		{
			std::cout << "List4: i: " << (i - q3)
					  << ", hi = " << sigs[i].h << ", si = " << sigs[i].s << std::endl;
		}
#endif
		/** save list*/
		std::cout << "master: saving split lists..." << std::endl;
		sigsave_it(sigs.begin(), sigs.begin() + q1, dir + L1_fname);
		sigsave_it(sigs.begin() + q1, sigs.begin() + q2, dir + L2_fname);
		sigsave_it(sigs.begin() + q2, sigs.begin() + q3, dir + L3_fname);
		sigsave_it(sigs.begin() + q3, sigs.end(), dir + L4_fname);

		/* Conversion: mpz_t -> uint64_t */
		std::cout << "master: converting mpz_t to uint64_t..." << std::endl;
		packhalf_it_opt(sigs.begin(), sigs.begin() + q1, L1_hhigh, L1_hlow, q1, ofst);
		packhalf_it_opt(sigs.begin() + q1, sigs.begin() + q2, L2_hhigh, L2_hlow, q2 - q1, ofst);
		packhalf_it_opt(sigs.begin() + q2, sigs.begin() + q3, L3_hhigh, L3_hlow, q3 - q2, ofst);
		packhalf_it_opt(sigs.begin() + q3, sigs.end(), L4_hhigh, L4_hlow, S - q3, ofst);
		std::cout << "L4_hhigh.size(): " << L4_hhigh.size() << std::endl;

		/**  c range of CSS2020*/
		// lower_threshold_uint = 0;
		// upper_threshold_uint = threshold_uint;

		/** optimal c rande */
		lower_threshold_uint = (a_bit + 1 <= threshold_bit) ? 0 : (1 << (a_bit)) - (threshold_uint >> 1) - 1;
		upper_threshold_uint = (a_bit + 1 <= threshold_bit) ? threshold_uint - 1 : (1 << (a_bit)) + (threshold_uint >> 1) - 1;
		std::cout << "lower_threshold_uint: " << lower_threshold_uint << ", upper_threshold_uint: " << upper_threshold_uint << std::endl;

		std::cout << "master: cleaning up the original GMP vector..." << std::endl;
		std::vector<SignatureSimple>().swap(sigs);
		malloc_trim(0); // required to clean up the orphaned memory allocated by GMP
	}

	uint32_t h_bit = l_bit - (ofst + pad - 1); // length of first h
	// uint32_t rshift_L = h_bit + 1 < 64 ? 0 : (h_bit + 1) - 64; // Bits to be shifted to the right to make it 64-bit when putting in L1p and L2p
	uint32_t pad_a_bit = h_bit - a_bit; // Number of bits to shift to the right to compute MSB_a(x)
	// uint32_t h_b_bit = h_bit + 1 - rshift_L; //Number of bits passed to (b) in algorithm
	uint32_t rshift_n = h_bit - ignore_bit; // the number of bits shifted to right to compute MSB_n(x)

	if (myrank == master)
	{
		std::cout << "h_bit: " << h_bit << std::endl;
		// std::cout << "rshift_L: " << rshift_L << std::endl;
		std::cout << "pad_a_bit: " << pad_a_bit << std::endl;
		// std::cout << "h_b_bit: " << h_b_bit << std::endl;
		std::cout << "rshift_n: " << rshift_n << std::endl;
	}
	std::cout << "bef barrier" << std::endl;
	// world.barrier();

	std::cout << "aft barrier" << std::endl;

	/* Broadcast sigs */
	// TODO: how to force broadcast to preallocate the exact memory for vector?
	// printf("[%d]/[%d]: distributing signature data..\n", myrank, worldsize);
	// broadcast(world, L1_hhigh, master);
	// broadcast(world, L1_hlow, master);
	// broadcast(world, L2_hhigh, master);
	// broadcast(world, L2_hlow, master);
	// broadcast(world, L3_hhigh, master);
	// broadcast(world, L3_hlow, master);
	// broadcast(world, L4_hhigh, master);
	// broadcast(world, L4_hlow, master);
	// printf("[%d]/[%d]: successfully distributed signature data!\n", myrank, worldsize);

	/* Conversion: uint64_t -> uint128_t */
	printf("[%d]/[%d]: converting uint64 -> uint128...\n", myrank, worldsize);
	uint64v_to_uint128v(L1_hhigh, L1_hlow, L1_h);
	uint64v_to_uint128v(L2_hhigh, L2_hlow, L2_h);
	uint64v_to_uint128v(L3_hhigh, L3_hlow, L3_h);
	uint64v_to_uint128v(L4_hhigh, L4_hlow, L4_h);
	printf("[%d]/[%d]: filled L1 with %lu signatures \n", myrank, worldsize, L1_h.size());
	printf("[%d]/[%d]: filled L2 with %lu signatures \n", myrank, worldsize, L2_h.size());
	printf("[%d]/[%d]: filled L3 with %lu signatures \n", myrank, worldsize, L3_h.size());
	printf("[%d]/[%d]: filled L4 with %lu signatures \n", myrank, worldsize, L4_h.size());

	/* Cleanup */
	printf("[%d]/[%d]: cleaning up uint64_t vectors...\n", myrank, worldsize);
	std::vector<uint64_t>().swap(L1_hhigh);
	std::vector<uint64_t>().swap(L1_hlow);
	std::vector<uint64_t>().swap(L2_hhigh);
	std::vector<uint64_t>().swap(L2_hlow);
	std::vector<uint64_t>().swap(L3_hhigh);
	std::vector<uint64_t>().swap(L3_hlow);
	std::vector<uint64_t>().swap(L4_hhigh);
	std::vector<uint64_t>().swap(L4_hlow);
	malloc_trim(0); // required to clean up the orphaned memory allocated by hish and low

	std::vector<Index> subresult;

	std::cout << "l_bit - (ofst + pad - 1) - a_bit: " << (l_bit - (ofst + pad - 1) - a_bit) << std::endl;
	std::cout << "(l_bit - ignore_bit - (ofst + pad - 1) ): " << (l_bit - ignore_bit - (ofst + pad - 1)) << std::endl;
#pragma omp parallel shared(index_result, L1_h, L2_h, L3_h, L4_h, ofst, pad, sig_percent)
	{
		bool break_flag = false;
		const int MARGIN_BIT = 0;
		uint128_t hi, hj, hk, hl, msb_a_x;
		std::vector<LRComb128> L1p, L2p;
		std::vector<Index> Lp;
		bool flag = true;

// MSB_a(x_1+x_2) = c and MSB_a(x_3+x_4) = c
#pragma omp for schedule(dynamic)
		for (uint128_t c = lower_threshold_uint; c < upper_threshold_uint; c++)
		// for (uint32_t c = 0; c < threshold_uint; c++)
		{
			if (break_flag)
			{
				continue;
			}
			if (flag)
			{
				std::cout << "small loop start: " << uint64_t(c) << std::endl;
				flag = false;
			}
			if (c % 2000 == 0)
			{
				std::cout << "small loop: " << uint64_t(c) << std::endl;
			}
			// printf("[%d]/[%d]: using %u threads\n", myrank, worldsize, n_threads);

#ifdef use_break_flag
			const uint64_t sub_margin = 20 * a_bit;
			sub_Lxp_max = calc_numLnP(a_bit, c + 1) + sub_margin;

#endif

			L1p.reserve(sub_Lxp_max);
			L2p.reserve(sub_Lxp_max);

#ifdef use_break_flag
			Lp_max = calc_numLP(a_bit, c, ignore_bit) + sub_margin * 2;
#endif

			Lp.reserve(Lp_max);

			// calc_add_MSB_pair(L1_h, L2_h, c, l_bit, a_bit, ofst, pad, L1p);
			// calc_add_MSB_pair(L3_h, L4_h, c, l_bit, a_bit, ofst, pad, L2p);
			for (uint32_t i = 0; i < L1_h.size(); i++)
			{
				if (break_flag)
					continue;
				hi = L1_h[i];
				uint128_t shifted_hi = hi >> pad_a_bit;
				if (shifted_hi > c) // shifted hi is no longer less than c, beacause the list is sorted.
				{
					// std::cout << "break: hi is ";
					// print_uint128(hi);
					break;
				}
				else if ((shifted_hi + a_bit_max_128) < c) // there is no hj which satisfies hi + hj = c*2^a
				{
					// std::cout << "continue: hi is ";
					// print_uint128(hi);
					continue;
				}

				if ((c << pad_a_bit) < hi + 1) // hi is already larger than c
				{
					// std::cout << "continue: ((c << (l_bit - a_bit)) < hi + 1) : ";
					continue;
				}

				std::vector<uint128_t>::iterator lower_bound_iterator = std::lower_bound(L2_h.begin(), L2_h.end(), ((c << pad_a_bit)) - hi - 1);
				for (uint32_t j = std::distance(L2_h.begin(), lower_bound_iterator); j < L2_h.size(); j++)
				{
					if (break_flag)
						continue;
					hj = L2_h[j];
					msb_a_x = ((hi + hj) >> (pad_a_bit));
					if (msb_a_x == c)
					{

#ifdef debug_four_list

						uint128_t msbhihj = (hi + hj) >> (l_bit - a_bit - (ofst + pad - 1));
						std::cout << "List1, c: " << uint64_t(c)
								  << ", hi: " << uint64_t(hi) << ", hj: " << uint64_t(hj)
								  << ", hi+hj: " << uint64_t((hi + hj)) << ", ((hi + hj) >> (l_bit - a_bit - (ofst + pad - 1))): " << uint64_t(msbhihj) << std::endl;
						// std::cout << "sub_index1: " << i << ", " << j << std::endl;
						// std::cout << "(uint64_t)(hi + hj): " << (uint64_t)(hi + hj) << std::endl;
#endif

						L1p.emplace_back((hi + hj), i, j);
#ifdef use_break_flag
						if (L1p.size() >= sub_Lxp_max)
						{
							break_flag = true;
							// std::cout <<"break_flag 1" << std::endl;
						}
#endif
					}
					else if (msb_a_x > c) // there is no hj, bacause hi + hj > c*2^a
					{
						// std::cout << "break hj: ";
						// print_uint128(hj);
						break;
					}
				}
			}

			break_flag = false;
			for (uint32_t k = 0; k < L3_h.size(); k++)
			{
				if (break_flag)
					continue;
				hk = L3_h[k];

				if ((hk >> (pad_a_bit)) > c) // Sorted, so no further exploration is pointless below c
				{
					break;
				}
				else if (((hk >> (pad_a_bit)) + a_bit_max_128) < c) // hk > c*2^a
				{
					continue;
				}
				if ((c << pad_a_bit) < hk + 1)
				{
					continue;
				}
				std::vector<uint128_t>::iterator lower_bound_iterator = std::lower_bound(L4_h.begin(), L4_h.end(), static_cast<uint128_t>((c << (pad_a_bit)) - hk - 1));
				for (uint32_t l = std::distance(L4_h.begin(), lower_bound_iterator); l < L4_h.size(); l++)
				{
					if (break_flag)
						continue;
					hl = L4_h[l];
					msb_a_x = ((hk + hl) >> pad_a_bit);
					if (msb_a_x == c)
					{

#ifdef debug_four_list

						uint128_t msbhkhl = (hk + hl) >> (l_bit - a_bit - (ofst + pad - 1));
						std::cout << "List2, c: " << uint64_t(c)
								  << ", hk: " << uint64_t(hk) << ", hl: " << uint64_t(hl)
								  << ", hk+hl: " << uint64_t((hk + hl)) << ", ((hk + hl) >> (l_bit - a_bit - (ofst + pad - 1))): " << uint64_t(msbhkhl) << std::endl;
						// std::cout << "sub_index1: " << i << ", " << j << std::endl;
						// std::cout << "(uint64_t)(hi + hj): " << (uint64_t)(hi + hj) << std::endl;
#endif

						L2p.emplace_back((hk + hl), k, l);
#ifdef use_break_flag
						if (L2p.size() >= sub_Lxp_max)
						{
							break_flag = true;
							// std::cout <<"break_flag 2" << std::endl;
						}
#endif
					}
					else if (msb_a_x > c) // hk+hl>c*2^a
					{
						break;
					}
				}
			}

			/** sort */
			spsort::integer_sort(L1p.begin(), L1p.end());
			spsort::integer_sort(L2p.begin(), L2p.end());

			// std::cout << "MSB minus: " << c << std::endl;
			// MSB_n(|x_1'-x_2'|) = 0
			break_flag = false;
			// std::cout << "L1p.size(): " << L1p.size() << ", L2p.size(): " << L2p.size() << std::endl;
			for (uint32_t i = 0; i < L1p.size(); i++)
			{
				if (break_flag)
					continue;
				hi = L1p[i].hsum;
				uint32_t lower_bound_index_L2p = 0;
				if (hi > L2p[0].hsum)
				{
					if (hi >= (uint128_t(1) << (rshift_n)))
					{ // *****
						lower_bound_index_L2p = lower_bound_serach(L2p, 0, L2p.size(), (hi - (uint128_t(1) << (rshift_n))));
					}
#ifdef debug_four_list
					// std::cout << "lower_bound_index_L2p: " << lower_bound_index_L2p << std::endl;
					// std::cout << "L2p[lower_bound_index_L2p].hsum: " << L2p[lower_bound_index_L2p].hsum << std::endl;
#endif
				}
				for (uint32_t j = lower_bound_index_L2p; j < L2p.size(); j++)
				{
#ifdef use_break_flag
					if (break_flag)
						continue;
#endif
					hj = L2p[j].hsum;
					if (hi >= hj)
					{

#ifdef debug_four_list
						// std::cout << "h: " << uint32_t(hi - hj) << std::endl;
						// std::cout << "((h) >> (l_bit - ignore_bit - (ofst + pad - 1) )): " << uint32_t((hi - hj) >> (l_bit - ignore_bit - (ofst + pad - 1))) << std::endl;
#endif

						if (((hi - hj) >> rshift_n) == 0)
						{

#ifdef debug_four_list
							// std::cout << "c: " << uint64_t(c) << ", hi: " << uint64_t(hi) << ", hj: " << uint64_t(hj) << ", hi-hj: " << uint64_t((hi - hj)) << std::endl;
							// std::cout <<"sum_Index1: "<<L1p[i].idx_L << ", "<< L1p[i].idx_R<<", " << L2p[j].idx_L <<", " << L2p[j].idx_R << std::endl;

#endif

							Lp.emplace_back(L1p[i].idx_L, L1p[i].idx_R, L2p[j].idx_L, L2p[j].idx_R, false);
#ifdef use_break_flag
							if (Lp.size() >= Lp_max)
							{
								break_flag = true;
								// std::cout <<"break_flag 3" << std::endl;
							}
#endif
						}
					}
					else if (hi < hj)
					{
#ifdef debug_four_list
						// std::cout << "-h: " << uint32_t(hj - hi) << std::endl;
						// std::cout <<"((-h) >> (l_bit - ignore_bit - (ofst + pad - 1) )): " << uint32_t((hj-hi) >> (l_bit - ignore_bit - (ofst + pad - 1) )) << std::endl;
#endif
						if (((hj - hi) >> (rshift_n)) == 0)
						{
#ifdef use_break_flag
							if (break_flag)
								continue;
#endif

#ifdef debug_four_list

							std::cout << "c: " << uint64_t(c) << ", hi: " << uint64_t(hi) << ", hj: " << uint64_t(hj) << ", -hi+hj: " << uint64_t((hj - hi)) << std::endl;
							// std::cout <<"sum_Index2: "<<L1p[i].idx_L << ", "<< L1p[i].idx_R<<", " << L2p[j].idx_L <<", " << L2p[j].idx_R << std::endl;
#endif
							Lp.emplace_back(L1p[i].idx_L, L1p[i].idx_R, L2p[j].idx_L, L2p[j].idx_R, true);
#ifdef use_break_flag
							if (Lp.size() >= Lp_max)
							{
								break_flag = true;
								// std::cout << "break_flag 4" << std::endl;
							}
#endif
						}
						if (hj > ((uint128_t(1) << (rshift_n)) + hi))
						{
							break;
						}
					}
				}
			}
			// std::cout << "small loop end: " << c << std::endl;
			// std::cout << "break_flag " << c << ": " << break_flag << std::endl;
			// std::cout << "c: " << uint64_t(c) << ", L1p.size(): " << L1p.size() << ", L2p.size(): " << L2p.size() << ", Lp.size(): " << Lp.size() << std::endl;
#pragma omp critical
			{
				// printf("[%d]/[%d]: pragma omp critical!\n", myrank, worldsize);
				// std::cout << "result.size(): " << result.size() << std::endl;
				// std::cout << "index_result.size(): " << index_result.size() << std::endl;
				// std::cout << "c: " << uint64_t(c) << ", Lp.size(): " << Lp.size() << std::endl;
				// std::cout << "c: " << uint64_t(c) << ", Lp.size(): " << Lp.size() << std::endl;
				if (Lp.size() != 0)
				{
					index_result.insert(index_result.end(), Lp.begin(), Lp.end());
				}

				if (index_result.size() > upper_sigs * sig_percent)
				{
					std::cout << "sig_percent: " << sig_percent << std::endl;
					sig_percent += 0.01;
				}
			}
			L1p.clear();
			L2p.clear();
			Lp.clear();
			L1p.shrink_to_fit();
			L2p.shrink_to_fit();
			Lp.shrink_to_fit();
			if (index_result.size() > upper_sigs)
			{
				std::cout << "index_result.size() > upper_sigs" << std::endl;
				break_flag = true;
			}
		}

		// std::vector<LRComb64>().swap(L1p);
		// std::vector<LRComb64>().swap(L2p);
		// std::vector<Index>().swap(Lp);
	} // pragma omp parallel

	printf("[%d]/[%d]: done!\n", myrank, worldsize);
	printf("[%d]/[%d]: freeing up memory..\n", myrank, worldsize);

	// recovery secret key form the indexes
	std::cout << "index_result:" << index_result.size() << std::endl;
	// world.barrier();
	if (myrank == master)
	{
		std::vector<SignatureSimple> L1, L2, L3, L4;
		L1.reserve(q1);
		L2.reserve(q2 - q1);
		L3.reserve(q3 - q2);
		L4.reserve(S - q3);
		printf("master: reloading the original samples from %s\n", (dir + L1_fname).c_str());
		sigload(L1, dir + L1_fname, q1);
		printf("master: reloading the original samples from %s\n", (dir + L2_fname).c_str());
		sigload(L2, dir + L2_fname, q2 - q1);
		printf("master: reloading the original samples from %s\n", (dir + L3_fname).c_str());
		sigload(L3, dir + L3_fname, q3 - q2);
		printf("master: reloading the original samples from %s\n", (dir + L4_fname).c_str());
		sigload(L4, dir + L4_fname, S - q3);
		printf("master: computing linear combinations from indices...\n");
		std::cout << "index_result.size(): " << index_result.size() << std::endl;
		restore_from_idx(sigs, index_result, L1, L2, L3, L4, (mpz_class)(1) << (l_bit - ignore_bit));

		if (sigs.size() < _keep_max)
		{
			printf("[%d]/[%d]: WARNING: found only %lu < %u collisions \n", myrank, worldsize, sigs.size(), _keep_max);
		}
		else
		{
			printf("[%d]/[%d]: found %lu collisions\n", myrank, worldsize, sigs.size());
		}
		printf("[%d]/[%d]: copying result..\n", myrank, worldsize);

		/* Save the result after each round */
		if (dir.length())
		{
			// file format: prefix_round-i.bin
			std::string outsig = out_prefix + "_round-" + std::to_string(round) + ".bin";
			printf("master: saving signatures of h < 2^%u to %s... \n", (l_bit - ignore_bit), (dir + outsig).c_str());
			sigsave(sigs, dir + outsig);
		}

		printf("master: cleaning up result, L1, L2, L3, L4...\n");
		// std::vector<Index>().swap(index_result);
		// std::vector<SignatureSimple>().swap(L1);
		// std::vector<SignatureSimple>().swap(L2);
		// std::vector<SignatureSimple>().swap(L3);
		// std::vector<SignatureSimple>().swap(L4);
		index_result.clear();
		L1.clear();
		L2.clear();
		L3.clear();
		L4.clear();
		index_result.shrink_to_fit();
		L1.shrink_to_fit();
		L2.shrink_to_fit();
		L3.shrink_to_fit();
		L4.shrink_to_fit();
	}
}

uint32_t lower_bound_serach(std::vector<LRComb128> &combs, const uint32_t &lower_index, const uint32_t &upper_index, const uint128_t &hj_cand)
{
	uint32_t left = lower_index;
	uint32_t right = upper_index;
	while (right - left > 1)
	{
		uint32_t mid = left + (right - left) / 2;
		if (combs[mid].hsum >= hj_cand)
		{
			right = mid;
		}
		else
		{
			left = mid;
		}
	}
	// std::cout <<"lower_index: " << lower_index <<", upper_index: " << upper_index <<", right: " << right <<", hj_cand: " << hj_cand << ", sigs[right].h: " <<sigs[right].h << std::endl;
	// if (right-1!=0){
	// 	std::cout << "sigs[right].h: " << sigs[right].h << std::endl;
	// 	std::cout << "sigs[right-1].h: " << sigs[right - 1].h << std::endl;
	// }
	return right;
}

void calc_add_MSB_pair(std::vector<uint128_t> &L1_h, std::vector<uint128_t> &L2_h, const uint128_t &c, const uint32_t &l_bit, const uint32_t a_bit, const int &ofst, const int &pad, std::vector<LRComb64> &outputList)
{
	bool break_flag = false;
	uint128_t hi, hj, msb_a_x;
	uint128_t a_bit_max_128 = (1LL << a_bit) - 1;

	const int MARGIN_BIT = 0;

	for (uint32_t i = 0; i < L1_h.size(); i++)
	{
		if (break_flag)
			continue;
		hi = L1_h[i];

#ifdef debug_four_list
		// std::cout << "log2(hi) :" << (std::log2(uint64_t(hi >> 64)) + 64) << std::endl;
		std::cout << "List1, c: " << uint64_t(c) << ", hi: " << uint64_t((hi)) << ", (hi >> (l_bit - (ofst + pad - 1) - a_bit + MARGIN_BIT)): " << uint32_t(hi >> (l_bit - (ofst + pad - 1) - a_bit + MARGIN_BIT)) << std::endl;
#endif

		if ((hi >> (l_bit - (ofst + pad - 1) - a_bit + MARGIN_BIT)) > c) // hi is already larger than c
		{
			// std::cout << "break: hi is " << hi << std::endl;
			break;
		}
		else if (((hi >> (l_bit - (ofst + pad - 1) - a_bit)) + a_bit_max_128) < c) // hi is already larger than c
		{
			// std::cout << "continue: hi is " << hi << std::endl;
			continue;
		}
		std::vector<uint128_t>::iterator lower_bound_iterator = std::lower_bound(L2_h.begin(), L2_h.end(), static_cast<uint128_t>((c << (l_bit - a_bit)) - hi - 1));
#ifdef debug_four_list
		// std::cout << "std::distance(L2_h.begin(), lower_bound_iterator): " << std::distance(L2_h.begin(), lower_bound_iterator) << std::endl;
		// std::cout << "L2_h[j]: "<<uint64_t((L2_h[std::distance(L2_h.begin(), lower_bound_iterator)])) << std::endl;
#endif

		for (uint64_t j = std::distance(L2_h.begin(), lower_bound_iterator); j < L2_h.size(); j++)
		{
			if (break_flag)
				continue;
			hj = L2_h[j];
			msb_a_x = ((hi + hj) >> (l_bit - (ofst + pad - 1) - a_bit));
			if (msb_a_x == c)
			{

				outputList.emplace_back((uint64_t)(hi + hj), i, j);
#ifdef debug_four_list

				uint128_t msbhihj = (hi + hj) >> (l_bit - a_bit - (ofst + pad - 1));
				std::cout << "List1, c: " << uint64_t(c)
						  << ", hi: " << uint64_t(hi) << ", hj: " << uint64_t(hj)
						  << ", hi+hj: " << uint64_t((hi + hj)) << ", ((hi + hj) >> (l_bit - a_bit - (ofst + pad - 1))): " << uint64_t(msbhihj) << std::endl;
				// std::cout << "sub_index1: " << i << ", " << j << std::endl;
				// std::cout << "(uint64_t)(hi + hj): " << (uint64_t)(hi + hj) << std::endl;
#endif
#ifdef use_break_flag
				if (outputList.size() >= sub_Lxp_max)
				{
					break_flag = true;
					// std::cout <<"break_flag 1" << std::endl;
				}
#endif
			}
			else if (msb_a_x > c) // hi + hj > c*2^l
			{
				// std::cout << "break2: " << msb_a_x << ", " << c << std::endl;
				break;
			}
		}
	}
}

uint64_t calc_part_sum(const uint32_t &start, const uint32_t &end)
{
	uint64_t s = (uint64_t)start;
	uint64_t e = (uint64_t)end;
	uint64_t sum = uint64_t((s + e) * (e - s + 1) / 2);
	return sum;
}

uint64_t calc_sum_square(const uint32_t &n)
{
	uint64_t N = (uint64_t)n;
	return uint64_t(N * (N + 1) * (2 * N + 1) / 6);
}

uint64_t calc_part_sum_square(const uint32_t &start, const uint32_t &end)
{
	uint64_t end_sum = calc_sum_square(end);
	uint64_t start_sum = calc_sum_square(start);

	if (start == end)
	{
		return (start + 1) * (start + 1);
	}
	else
	{
		return end_sum - start_sum;
	}
}

uint64_t calc_numLnP(const uint32_t &a, const uint32_t &c)
{
	uint64_t a_num = (1UL << a);
	uint64_t margin = 1;
	if (c < a_num)
	{
		return uint64_t(c + 1) + margin;
	}
	else
	{
		return uint64_t((1UL << (a + 1)) - 2 - ((uint64_t)c) + margin);
	}
}

uint64_t calc_numLP(const uint32_t &a, const uint32_t &c, const uint32_t &n)
{
	uint64_t c_ = (uint64_t)c;
	uint64_t numLnP = calc_numLnP(a, c);
	uint64_t comb = numLnP * numLnP;
	double comb_pow = std::log2(comb);
	int comb_int_pow = std::floor(comb_pow);
	double comb_double_pow = comb_pow - comb_int_pow;
	int comb_int_pow_ceil = std::ceil(comb_pow);

	uint64_t ans = 100;
	if (comb_int_pow >= (n - a))
	{
		ans = (1UL << (comb_int_pow_ceil - n + a));
		// ans = (1UL << (comb_int_pow - n + a)) * std::ceil(std::pow(2, comb_double_pow));
	}
	// std::cout << "ans: " << ans << std::endl;

	return ans;
}

uint64_t calc_num_out(const uint32_t &a, const uint32_t &v, const uint32_t &n)
{
	uint32_t start = (1U << a) - (1U << (v - 1));
	uint32_t end = (1U << a) - 2;
	uint64_t term1 = 2 * (calc_part_sum_square(start, end));
	uint64_t term2 = (1UL << (2 * a)) + (1UL << v) - (1UL << (a + v));
	uint64_t term3 = (1UL << (2 * a));
	uint64_t term_sum = term1 + term2 + term3;
	double term_sum_pow = std::log2(term_sum);
	uint64_t int_pow = std::floor(term_sum_pow);
	double double_pow = term_sum_pow - int_pow;
	uint64_t out_int_pow = int_pow + a - n;
	uint64_t num_out_a = (1UL << out_int_pow) + std::ceil(std::pow(2, double_pow));
	uint64_t num_out = (num_out_a << 2);
	return num_out;
}

void schroeppel_shamir_mpi(std::vector<SignatureSimple> &sigs, mpz_class &bound, const uint32_t n_bit, uint32_t l, uint32_t b,
						   const uint32_t filter, uint32_t a, const int log_prec,
						   const std::vector<uint32_t> &b_info, const std::vector<uint32_t> &l_info, const size_t iota = 1,
						   const std::string out_prefix = "", const bool out_index = false, std::string dir = "", const bool istest = true)
{
	mpi::environment env;
	mpi::communicator world;
	const int master = 0;
	const int myrank = world.rank();
	const int worldsize = world.size();

	uint32_t threshold_bit = n_bit - filter;   // threshold bit for the next samples
	uint32_t c_threshold_bit = n_bit - filter; // threshold bit for the current samples
	mpz_class threshold_mpz;
	uint32_t keep_min;
	uint32_t keep_max;
	uint32_t l_next;
	uint32_t S, q1, q2, q3;
	const std::string fname_prefix = "ss-mpi";
	if ((b_info.size() != iota) | (l_info.size() != iota + 1))
	{
		printf("ERROR: invalid offset_info/b_info/l_info\n");
		return;
	}
	if (dir.length())
	{
		dir = dir + "/";
	}

	for (size_t round = 0; round < iota; round++)
	{
		if (myrank == master)
			printf("-------------------------------------------------- Round %lu begins --------------------------------------------------\n", round);
		/* Compute how many samples to be kept */
		if (l_info.empty())
		{
			keep_min = 1 << l;
			keep_max = keep_min * 2; // tentative
			a = l - 2;
		}
		else
		{
			l_next = l_info[round + 1];
			keep_min = 1 << (l_next);
			if (round == 0)
			{
				keep_max = keep_min * 1.01; // tentative
			}
			else if (round == 1)
			{
				keep_max = keep_min * 1.01;
			}
			else
			{
				keep_max = keep_min * 1.01;
			}

			a = l_info[round] - 2;
		}

		/* Compute the next threshold value */
		c_threshold_bit = threshold_bit;
		if (b_info.empty())
			threshold_bit -= b;
		else
			threshold_bit -= b_info[round];

		threshold_mpz = mpz_class(1) << threshold_bit;
		// Note that b has to be larger than a
		// Low word has to be strictly smaller than this bound
		uint64_t threshold_64 = 1ULL << (64 - (b_info[round] - a));
		int ofst;
		size_t pad;
		compute_ofst_uint128(ofst, pad, c_threshold_bit, 1);

		if (myrank == master)
		{
			printf("master: b=%u, a=%u\n", b_info[round], a);
			printf("master: ofst = %d, pad = %lu\n", ofst, pad);
			printf("master: c_threshold_bit = %u\n", c_threshold_bit);
			printf("master: threshold_bit = %u\n", threshold_bit);
			printf("master: threshold_64 = %lu\n", threshold_64);
			gmp_printf("master: threshold_mpz = %Zd\n", threshold_mpz.get_mpz_t());
		}

		// File format: ss-mpi_round-i_{L1,R1,L2,R2}.bin
		std::string L1_fname, R1_fname, L2_fname, R2_fname;
		L1_fname = fname_prefix + "_round-" + std::to_string(round) + "_L1.bin";
		R1_fname = fname_prefix + "_round-" + std::to_string(round) + "_R1.bin";
		L2_fname = fname_prefix + "_round-" + std::to_string(round) + "_L2.bin";
		R2_fname = fname_prefix + "_round-" + std::to_string(round) + "_R2.bin";

		std::vector<uint128_t> L1_h, R1_h, L2_h, R2_h;
		std::vector<uint64_t> L1_hhigh, R1_hhigh, L2_hhigh, R2_hhigh, L1_hlow, R1_hlow, L2_hlow, R2_hlow;
		if (myrank == master)
		{
			/* Split sigs into L1 || R1 || L2 || R2 */
			S = sigs.size();
			q1 = S / 4;
			q2 = S / 2;
			q3 = S * 3 / 4;
			std::cout << "sigs.size(): " << S << std::endl;
#if 1
			std::cout << "master: sorting L1 in descending order..." << std::endl;
			std::sort(sigs.begin(), sigs.begin() + q1, std::greater<SignatureSimple>());
			std::cout << "master: sorting R1 in ascending order..." << std::endl;
			std::sort(sigs.begin() + q1, sigs.begin() + q2);
			std::cout << "master: sorting L2 in descending order..." << std::endl;
			std::sort(sigs.begin() + q2, sigs.begin() + q3, std::greater<SignatureSimple>());
			std::cout << "master: sorting R2 in ascending order..." << std::endl;
			std::sort(sigs.begin() + q3, sigs.end());
			std::cout << "master: sorting done" << std::endl;
#endif
			// std::cin.ignore();

			/* Save split sigs */
			if (!istest)
			{
				std::cout << "master: saving split lists..." << std::endl;
				sigsave_it(sigs.begin(), sigs.begin() + q1, dir + L1_fname);
				sigsave_it(sigs.begin() + q1, sigs.begin() + q2, dir + R1_fname);
				sigsave_it(sigs.begin() + q2, sigs.begin() + q3, dir + L2_fname);
				sigsave_it(sigs.begin() + q3, sigs.end(), dir + R2_fname);
				// std::cin.ignore();
			}

			/* Conversion: mpz_t -> uint64_t */
			std::cout << "master: converting mpz_t to uint64_t..." << std::endl;
			packhalf_it_opt(sigs.begin(), sigs.begin() + q1, L1_hhigh, L1_hlow, q1, ofst);
			packhalf_it_opt(sigs.begin() + q1, sigs.begin() + q2, R1_hhigh, R1_hlow, q2 - q1, ofst);
			packhalf_it_opt(sigs.begin() + q2, sigs.begin() + q3, L2_hhigh, L2_hlow, q3 - q2, ofst);
			packhalf_it_opt(sigs.begin() + q3, sigs.end(), R2_hhigh, R2_hlow, S - q3, ofst);
			// std::cin.ignore();
#if 1
			std::cout << "master: cleaning up the original GMP vector..." << std::endl;
			std::vector<SignatureSimple>().swap(sigs);
			malloc_trim(0); // required to clean up the orphaned memory allocated by GMP
#endif
		}
		world.barrier();
		// std::cin.ignore();

		/* Broadcast sigs */
		// TODO: how to force broadcast to preallocate the exact memory for vector?
		printf("[%d]/[%d]: distributing signature data..\n", myrank, worldsize);
		broadcast(world, L1_hhigh, master);
		;
		broadcast(world, L1_hlow, master);
		;
		broadcast(world, R1_hhigh, master);
		broadcast(world, R1_hlow, master);
		broadcast(world, L2_hhigh, master);
		broadcast(world, L2_hlow, master);
		broadcast(world, R2_hhigh, master);
		broadcast(world, R2_hlow, master);
		printf("[%d]/[%d]: successfully distributed signature data!\n", myrank, worldsize);

		/* Conversion: uint64_t -> uint128_t */
		printf("[%d]/[%d]: converting uint64 -> uint128...\n", myrank, worldsize);
		uint64v_to_uint128v(L1_hhigh, L1_hlow, L1_h);
		uint64v_to_uint128v(R1_hhigh, R1_hlow, R1_h);
		uint64v_to_uint128v(L2_hhigh, L2_hlow, L2_h);
		uint64v_to_uint128v(R2_hhigh, R2_hlow, R2_h);
		printf("[%d]/[%d]: filled L1 with %lu signatures \n", myrank, worldsize, L1_h.size());
		printf("[%d]/[%d]: filled R1 with %lu signatures \n", myrank, worldsize, R1_h.size());
		printf("[%d]/[%d]: filled L2 with %lu signatures \n", myrank, worldsize, L2_h.size());
		printf("[%d]/[%d]: filled R2 with %lu signatures \n", myrank, worldsize, R2_h.size());

		/* Cleanup */
		printf("[%d]/[%d]: cleaning up uint64_t vectors...\n", myrank, worldsize);
		std::vector<uint64_t>().swap(L1_hhigh);
		std::vector<uint64_t>().swap(L1_hlow);
		std::vector<uint64_t>().swap(R1_hhigh);
		std::vector<uint64_t>().swap(R1_hlow);
		std::vector<uint64_t>().swap(L2_hhigh);
		std::vector<uint64_t>().swap(L2_hlow);
		std::vector<uint64_t>().swap(R2_hhigh);
		std::vector<uint64_t>().swap(R2_hlow);
		// std::cin.ignore();

		/* Job scheduling */
		const uint32_t A_lim = 1U << a;
		const uint32_t range = (A_lim + worldsize - 1) / worldsize;

		if (myrank == master)
		{
			printf("master: trying to get 2^%u values less than 2^%u\n"
				   "[keep_min, keep_max] = [%u, %u]\n",
				   l_next, threshold_bit, keep_min, keep_max);
			printf("master: looking for collisions on top %u-bits, range is %u\n", a, range);
		}

		/* Start measuring time */
		printf("[%d]/[%d]: reduction started\n", myrank, worldsize);
		auto start = std::chrono::high_resolution_clock::now();

		float percent = 0;
		std::vector<Index> subresult;
		const uint32_t keep_min_sub = keep_min / worldsize;
		const uint32_t keep_max_sub = keep_max / worldsize;
		// Don't change this! The scaling factor below affects total memory a lot with many threads
		const uint32_t keep_max_combs = (1 << a) * 1.05;
		/*
		 * Decide how many collisions to be kept per round based on the tradeoff formula.
		 * 		m' = a' + 2 =  3*a + v - n
		 * Hence for v = 0 (i.e., one iteration of HGJ) m' = 3*a - n collisions are expected.
		 * If n = 3*a - 2 then m' = 4 collisions are expected per each iteration.
		 * If this is repeated for each [0, 2^a) then we get 2^m' = 2^a * 4 samples again.
		 * In practice we get even more and that's why the bound below is multiplied a bit.
		 * */
		uint64_t keep_max_combs_A = (1UL << (3 * a - b_info[round])) * 16;
		uint32_t n_threads = omp_get_max_threads();
		if (keep_max_combs_A > keep_max_sub / n_threads)
			keep_max_combs_A = keep_max_sub / n_threads;
		printf("[%d]/[%d]: using %u threads\n", myrank, worldsize, n_threads);
		if (myrank == master)
			printf("[%d]/[%d]: keep_max_combs_A = %lu \n", myrank, worldsize, keep_max_combs_A);
		const uint32_t log_mod = keep_min_sub / log_prec;
		size_t log_th = log_mod;
		const float percent_delta = 100.0 / log_prec;
		subresult.reserve(keep_max_sub * 1.1); // we need some margin because multiple threads may push partial results at a time
#pragma omp parallel shared(subresult, L1_h, R1_h, L2_h, R2_h, threshold_64, c_threshold_bit, threshold_bit, ofst, pad, log_th, percent)
		{
			std::vector<LRComb64> combs1;
			std::vector<LRComb64> combs2;
			std::vector<Index> combs_A;
			combs1.reserve(keep_max_combs);
			combs2.reserve(keep_max_combs);
			combs_A.reserve(keep_max_combs_A);
			uint64_t before = 0, after = 0, ubefore = 0, uafter = 0;
#pragma omp for schedule(dynamic)
			for (uint32_t k = 0; k < range; k++)
			{
				for (int rev = 0; rev < 2; rev++)
				{
#if 0
					ubefore = rdtsc();
#endif
					uint32_t A = worldsize * k + myrank;
					if (A >= A_lim || subresult.size() >= keep_max_sub)
						continue;

					if (rev == 0)
					{
						A = A_lim + A;
					}
					else
					{
						A = A_lim - 1 - A;
					}
					// printf("[%d]/[%d]: Finding collisions with A = %u\n", myrank, worldsize, A);

					/* Look for collisions on top a-bits */
#if 0
					before = rdtsc();
					collect_lc_two(combs1, L1_h, R1_h, A, a, c_threshold_bit, ofst, pad, keep_max_combs);
					after = rdtsc();
					printf("%lu cycles @collect_lc_two-1:\n", after-before);
#else
					collect_lc_two(combs1, L1_h, R1_h, A, a, c_threshold_bit, ofst, pad, keep_max_combs);
#endif
					if (combs1.size() == 0)
						continue;
#if 0
					before = rdtsc();
					//std::sort(combs1.begin(), combs1.end());
					spsort::integer_sort(combs1.begin(), combs1.end());
					after = rdtsc();
					printf("%lu cycles @sort-1\n", after-before);
#else
					spsort::integer_sort(combs1.begin(), combs1.end());
#endif

#if 0
					before = rdtsc();
					collect_lc_two(combs2, L2_h, R2_h, A, a, c_threshold_bit, ofst, pad, keep_max_combs);
					after = rdtsc();
					printf("%lu cycles @collect_lc_two-2:\n", after-before);
#else
					collect_lc_two(combs2, L2_h, R2_h, A, a, c_threshold_bit, ofst, pad, keep_max_combs);
#endif
					if (combs2.size() == 0)
						continue;
#if 0
					before = rdtsc();
					//std::sort(combs2.begin(), combs2.end());
					spsort::integer_sort(combs2.begin(), combs2.end());
					after = rdtsc();
					printf("%lu cycles @sort-2\n", after-before);
#else
					spsort::integer_sort(combs2.begin(), combs2.end());
#endif
#if 0
					printf("[%d]/[%d]: A = %u: Found (%lu, %lu) partial collisions\n",
							myrank, worldsize, A, combs1.size(), combs2.size());
#endif
#if 0
					printf( "[%d]/[%d]@%s: thread %d checks A = %u.\n",
							myrank, worldsize, env.processor_name().c_str(), omp_get_thread_num (), A);
#endif
					uint32_t i = 0;
					uint32_t j = 0;
					bool flip;

#if 0
					before = rdtsc();
#endif
					// std::cout <<"combs1.size(): " << combs1.size() << ", combs2.size(): " << combs2.size() << std::endl;
					int h_diff_count = 0;
					while (i < combs1.size() && j < combs2.size())
					{
						uint64_t lsum = combs1[i].hsum;
						uint64_t rsum = combs2[j].hsum;
						flip = lsum < rsum;
						uint64_t h_diff;
						if (flip)
							h_diff = rsum - lsum;
						else
							h_diff = lsum - rsum;
						h_diff_count += 1;
						if (h_diff_count < 100)
						{
							// std::cout << "combs_A.size(): " << combs_A.size() << ", keep_max_combs_A: " << keep_max_combs_A << std::endl;
							// std::cout << "h_diff: " << h_diff << ", threshold_64: " << threshold_64 << std::endl;
						}
						if (h_diff < threshold_64 && combs_A.size() < keep_max_combs_A)
						{
							combs_A.emplace_back(combs1[i].idx_L, combs1[i].idx_R, combs2[j].idx_L, combs2[j].idx_R,
												 flip);
						}
						if (flip)
							i++;
						else
							j++;
					}
#if 0
					after = rdtsc();
					printf("%lu cycles @while-loop\n", after-before);
#endif
#if 0
					printf("[%d]/[%d]: A = %u: Found %lu total collisions\n", myrank, worldsize, A, combs_A.size());
#endif
#pragma omp critical
					{
						// std::cout << "subresult.insert() "<< subresult.size() <<", combs_A.size(): " << combs_A.size() << std::endl;
						subresult.insert(subresult.end(), combs_A.begin(), combs_A.end());
						if (subresult.size() >= log_th)
						{
							uint32_t incr = (subresult.size() - log_th) / log_mod + 1;
							percent += percent_delta * incr;
							log_th += log_mod * incr;
							printf("[%d]/[%d]: %.2f %% done\n", myrank, worldsize, percent);
						}
					}
					combs1.resize(0);
					combs2.resize(0);
					combs_A.resize(0);
					combs1.reserve(keep_max_combs);
					combs2.reserve(keep_max_combs);
					combs_A.reserve(keep_max_combs_A);
#if 0
					uafter = rdtsc();
					printf("%lu cycles @one-loop\n", uafter-ubefore);
#endif
				} // for rev
			}	  // for k
			printf("[%d]/[%d]@%s: thread %d is freeing up memory...\n",
				   myrank, worldsize, env.processor_name().c_str(), omp_get_thread_num());
			std::vector<LRComb64>().swap(combs1);
			std::vector<LRComb64>().swap(combs2);
			std::vector<Index>().swap(combs_A);
		} // #pragma omp parallel shared
		if (subresult.size() < keep_min_sub)
		{
			printf("[%d]/[%d]: WARNING: got %lu subresult < keep_min_sub = %u; failed to get sufficiently many collisions!\n",
				   myrank, worldsize, subresult.size(), keep_min_sub);
		}
		if (subresult.size() > keep_max_sub)
		{
			printf("[%d]/[%d]: got %lu subresult > keep_max_sub = %u; truncating.\n",
				   myrank, worldsize, subresult.size(), keep_max_sub);
			subresult.resize(keep_max_sub);
		}
		printf("[%d]/[%d]: got %lu subresult\n", myrank, worldsize, subresult.size());

		/* End measuring time */
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		printf("[%d]/[%d]: Elapsed time for round %lu = %1.f seconds\n", myrank, worldsize, round, elapsed.count());

		/*
		 * Save interim report (only indices).
		 * file format: index_round-i_rank-j.bin
		 * Turn this on when launching the large attack.
		 * In case of MPI communication failure in the following steps we can still manually
		 * recover the linear combinations from index files.
		 */
		if (out_index == true)
		{
			std::string outidx = "index_round-" + std::to_string(round) + "_rank-" + std::to_string(myrank) + ".bin";
			printf("[%d]/[%d]: Saving Index subresult to %s...; leading to h < 2^%u \n", myrank, worldsize, outidx.c_str(), threshold_bit);
			idxsave(subresult, dir + outidx);
			/* debug logging */
			/*
			 std::vector<Index>().swap(subresult);
			 idxload(subresult, outidx);
			 printf("(%u, %u, %u, %u), flip=%d \n", subresult[0].idx_L1, subresult[0].idx_R1, subresult[0].idx_L2, subresult[0].idx_R2, subresult[0].flip);
			 */
		}

		/* Cleanup */
		printf("[%d]/[%d]: cleaning up the original lists L1_h, R1_h, L2_h, R2_h... \n", myrank, worldsize);
		std::vector<uint128_t>().swap(L1_h);
		std::vector<uint128_t>().swap(R1_h);
		std::vector<uint128_t>().swap(L2_h);
		std::vector<uint128_t>().swap(R2_h);
		printf("[%d]/[%d]: waiting for other nodes to finish... \n", myrank, worldsize);
		world.barrier();

		/* Prepare subres_sizes for gatherv */
		printf("[%d]/[%d]: preparing subres_sizes for gatherv... \n", myrank, worldsize);
		std::vector<int> subres_sizes;
		gather(world, static_cast<int>(subresult.size()), subres_sizes, master);
		// casting is required since MPI gatherv does not allow size specification with unsigned int...

		/* Convert Index to vector */
		printf("[%d]/[%d]: converting Index struct to separate vectors... \n", myrank, worldsize);
		std::vector<uint32_t> subresult_L1, subresult_R1, subresult_L2, subresult_R2;
		std::vector<uint8_t> subresult_fl;
		subresult_L1.reserve(subresult.size());
		subresult_R1.reserve(subresult.size());
		subresult_L2.reserve(subresult.size());
		subresult_R2.reserve(subresult.size());
		subresult_fl.reserve(subresult.size());
		for (auto &I : subresult)
		{
			subresult_L1.emplace_back(I.idx_L1);
			subresult_R1.emplace_back(I.idx_R1);
			subresult_L2.emplace_back(I.idx_L2);
			subresult_R2.emplace_back(I.idx_R2);
			subresult_fl.emplace_back(static_cast<uint8_t>(I.flip));
		}
		printf("[%d]/[%d]: cleaning up subresult... \n", myrank, worldsize);
		std::vector<Index>().swap(subresult);

		/* gatherv subresult into result*/
		std::vector<uint32_t> result_L1, result_R1, result_L2, result_R2;
		std::vector<uint8_t> result_fl;
		if (myrank == master)
		{
			printf("master: start gathering results...\n");
			uint32_t res_size = 0;

			for (auto &subsize : subres_sizes)
				res_size += subsize;

			result_L1.resize(res_size);
			result_R1.resize(res_size);
			result_L2.resize(res_size);
			result_R2.resize(res_size);
			result_fl.resize(res_size);

			printf("master: gathering subresults...\n");
			gatherv(world, subresult_L1, &result_L1[0], subres_sizes, master);
			printf("master: gathered %lu elements in result_L1\n", result_L1.size());
			gatherv(world, subresult_R1, &result_R1[0], subres_sizes, master);
			printf("master: gathered %lu elements in result_R1\n", result_R1.size());
			gatherv(world, subresult_L2, &result_L2[0], subres_sizes, master);
			printf("master: gathered %lu elements in result_L2\n", result_L2.size());
			gatherv(world, subresult_R2, &result_R2[0], subres_sizes, master);
			printf("master: gathered %lu elements in result_R2\n", result_R2.size());
			gatherv(world, subresult_fl, &result_fl[0], subres_sizes, master);
			printf("master: gathered %lu elements in result_fl\n", result_fl.size());
			printf("master: gatherv completed successfully!\n");
		}
		else
		{
			printf("[%d]/[%d]: sending subresult... \n", myrank, worldsize);
			gatherv(world, subresult_L1, master);
			gatherv(world, subresult_R1, master);
			gatherv(world, subresult_L2, master);
			gatherv(world, subresult_R2, master);
			gatherv(world, subresult_fl, master);
		}
		printf("[%d]/[%d]: cleaning up subresult vectors... \n", myrank, worldsize);
		std::vector<int>().swap(subres_sizes);

		std::vector<uint32_t>().swap(subresult_L1);
		std::vector<uint32_t>().swap(subresult_R1);
		std::vector<uint32_t>().swap(subresult_L2);
		std::vector<uint32_t>().swap(subresult_R2);
		std::vector<uint8_t>().swap(subresult_fl);

		if (istest)
		{
			printf("master: reduction ended; exiting test mode.\n");
			return;
		}

		/* Fukugen */
		std::vector<Index> result;
		if (myrank == master)
		{
			printf("master: reconstructing Index struct...\n");
			result.reserve(result_L1.size());
			for (size_t i = 0; i < result_L1.size(); i++)
			{
				result.emplace_back(result_L1[i], result_R1[i], result_L2[i], result_R2[i],
									static_cast<bool>(result_fl[i]));
			}
			printf("master: gathered %lu elements in result\n", result.size());
			printf("master: cleaning up separate result vectors...\n");
			std::vector<uint32_t>().swap(result_L1);
			std::vector<uint32_t>().swap(result_R1);
			std::vector<uint32_t>().swap(result_L2);
			std::vector<uint32_t>().swap(result_R2);
			std::vector<uint8_t>().swap(result_fl);

			std::vector<SignatureSimple> L1, R1, L2, R2;
			L1.reserve(q1);
			R1.reserve(q2 - q1);
			L2.reserve(q3 - q2);
			R2.reserve(S - q3);
			printf("master: reloading the original samples from %s\n", (dir + L1_fname).c_str());
			sigload(L1, dir + L1_fname, q1);
			printf("master: reloading the original samples from %s\n", (dir + R1_fname).c_str());
			sigload(R1, dir + R1_fname, q2 - q1);
			printf("master: reloading the original samples from %s\n", (dir + L2_fname).c_str());
			sigload(L2, dir + L2_fname, q3 - q2);
			printf("master: reloading the original samples from %s\n", (dir + R2_fname).c_str());
			sigload(R2, dir + R2_fname, S - q3);
			sigs.reserve(result.size());
			printf("master: computing linear combinations from indices...\n");
			restore_from_idx(sigs, result, L1, R1, L2, R2, threshold_mpz);

			printf("master: got %lu result \n", sigs.size());
			if (sigs.size() < keep_min)
				puts("WARNING: failed to get expected amount of reduced values!");

			/* Save the result after each round */
			if (out_prefix.length())
			{
				// file format: prefix_round-i.bin
				std::string outsig = out_prefix + "_round-" + std::to_string(round) + ".bin";
				printf("master: saving signatures of h < 2^%u to %s... \n", threshold_bit, (dir + outsig).c_str());
				sigsave(sigs, dir + outsig);
			}

			/* cleanups */
			printf("master: cleaning up result, L1, R1, L2, R2...\n");
			std::vector<Index>().swap(result);
			std::vector<SignatureSimple>().swap(L1);
			std::vector<SignatureSimple>().swap(R1);
			std::vector<SignatureSimple>().swap(L2);
			std::vector<SignatureSimple>().swap(R2);
			malloc_trim(0);
			printf("-------------------------------------------------- Round %lu finished --------------------------------------------------\n", round);
		}
	} // for round

	if (myrank == master)
	{
		bound = threshold_mpz;
		gmp_printf("master: reduction ended after %u rounds, h < %Zd\n", iota, bound.get_mpz_t());
	}
}

void restore_from_idx(std::vector<SignatureSimple> &sigs, const std::vector<Index> &idxlist,
					  const std::vector<SignatureSimple> &L1, const std::vector<SignatureSimple> &R1,
					  const std::vector<SignatureSimple> &L2, const std::vector<SignatureSimple> &R2,
					  const mpz_class &threshold_mpz)
{
	std::cout << "threshold_mpz: " << threshold_mpz << std::endl;
	uint64_t count = 0;
	float percent = 0.1;
	for (auto &r : idxlist)
	{
		// if(r.idx_L1 >= L1.size()){
		// 	std::cout << "out of index L1: " << r.idx_L1 << std::endl;
		// }
		// if(r.idx_R1 >= R1.size()){
		// 	std::cout << "out of index R1: " << r.idx_R1 << std::endl;
		// }
		// if(r.idx_L2 >= L2.size()){
		// 	std::cout << "out of index L2: " << r.idx_L2 << std::endl;
		// }
		// if(r.idx_R2 >= R2.size()){
		// 	std::cout << "out of index R2: " << r.idx_R2 << std::endl;
		// }

		count++;
		if (idxlist.size() * percent == count)
		{
			std::cout << "recovery percent: " << percent << std::endl;
			percent += 0.1;
		}
		mpz_class h = L1[r.idx_L1].h + R1[r.idx_R1].h - L2[r.idx_L2].h - R2[r.idx_R2].h;
		mpz_class s = L1[r.idx_L1].s + R1[r.idx_R1].s - L2[r.idx_L2].s - R2[r.idx_R2].s;

#ifdef debug_four_list
		std::cout << "index: " << r.idx_L1 << ", " << r.idx_R1 << ", " << r.idx_L2 << ", " << r.idx_R2 << ", h: " << h << ", s: " << s << std::endl;
#endif
		if (abs(h) >= threshold_mpz)
		{
			std::cout << "h is " << h << ". each value: " << L1[r.idx_L1].h << ", " << R1[r.idx_R1].h << ", " << L2[r.idx_L2].h << ", " << R2[r.idx_R2].h << std::endl;
			std::cout << "r.idx_R2: " << r.idx_R2 << std::endl;
			printf("WARNING: found h of %lu-bit at (%u, %u, %u, %u), skipping\n",
				   mpz_sizeinbase(h.get_mpz_t(), 2), r.idx_L1, r.idx_R1, r.idx_L2, r.idx_R2);
			continue;
		}
		if (h < 0)
		{
			if (!r.flip)
			{
				printf("WARNING: flip info was incorrect; recovering\n");
				mpz_class leftsum = L1[r.idx_L1].h + R1[r.idx_R1].h;
				mpz_class rightsum = L2[r.idx_L2].h + R2[r.idx_R2].h;
				// gmp_printf("lsum = %Zd\n", leftsum.get_mpz_t());
				// gmp_printf("rsum = %Zd\n", rightsum.get_mpz_t());
			}
			sigs.emplace_back(-h, -s);
		}
		else
		{
			if (r.flip)
			{
				printf("WARNING: flip info was incorrect; recovering\n");
				mpz_class leftsum = L1[r.idx_L1].h + R1[r.idx_R1].h;
				mpz_class rightsum = L2[r.idx_L2].h + R2[r.idx_R2].h;
				// gmp_printf("lsum = %Zd\n", leftsum.get_mpz_t());
				// gmp_printf("rsum = %Zd\n", rightsum.get_mpz_t());
			}
			sigs.emplace_back(h, s);
		}
	}
}

/* Old SS code without parallelization. */
void schroeppel_shamir(std::vector<SignatureSimple> *sigsptr, const uint32_t n_bit, const uint32_t l, const uint32_t b,
					   const uint32_t filter, const size_t iota = 1)
{
	uint32_t threshold_bit = n_bit - filter;
	uint32_t S = sigsptr->size();
	uint32_t keep = 1 << l;
	std::vector<SignatureSimple> *L1;
	std::vector<SignatureSimple> *R1;
	std::vector<SignatureSimple> *L2;
	std::vector<SignatureSimple> *R2;

	printf("schroeppel_shamir: got %u sigs\n", S);

	for (size_t round = 0; round < iota; round++)
	{
		S = sigsptr->size();

		/* split sigs into L1 || R1 || L2 || R2 */
		uint32_t q1 = S / 4;
		uint32_t q2 = S / 2;
		uint32_t q3 = S * 3 / 4;

		std::cout << "Splitting into 4 lists..." << std::endl;
		L1 = new std::vector<SignatureSimple>(sigsptr->begin(), sigsptr->begin() + q1);
		R1 = new std::vector<SignatureSimple>(sigsptr->begin() + q1, sigsptr->begin() + q2);
		L2 = new std::vector<SignatureSimple>(sigsptr->begin() + q2, sigsptr->begin() + q3);
		R2 = new std::vector<SignatureSimple>(sigsptr->begin() + q3, sigsptr->end());
		delete sigsptr;
		sigsptr = new std::vector<SignatureSimple>();

		threshold_bit -= b;

		mpz_class threshold;
		mpz_ui_pow_ui(threshold.get_mpz_t(), 2, threshold_bit);

		/* Init heaps */
		std::cout << "Sorting R1 and R2..." << std::endl;
		std::sort(R1->begin(), R1->end());
		std::sort(R2->begin(), R2->end());

		std::cout << "Initializing heaps..." << std::endl;
		std::priority_queue<LRComb, std::vector<LRComb>> heap1;
		std::priority_queue<LRComb, std::vector<LRComb>> heap2;
		for (uint32_t i = 0; i != L1->size(); i++)
		{
			heap1.push(LRComb((*L1)[i].h + (*R1)[0].h, i, 0));
			heap2.push(LRComb((*L2)[i].h + (*R2)[0].h, i, 0));
		}

		/* Start looking for collisions */
		std::cout << "Trying to get 2^" << l << " values less than 2^" << threshold_bit << std::endl;
		uint32_t counter = 0;
		uint32_t num_col = 0;
		mpz_class hsum1, hsum2, h_diff, s_diff;
		uint32_t i1, j1, i2, j2;
		while (num_col < keep && heap1.empty() == false && heap2.empty() == false)
		{
			counter++;
			hsum1 = heap1.top().hsum;
			i1 = heap1.top().idx_L;
			j1 = heap1.top().idx_R;

			hsum2 = heap2.top().hsum;
			i2 = heap2.top().idx_L;
			j2 = heap2.top().idx_R;

			h_diff = hsum1 - hsum2;
			s_diff = (*L1)[i1].s + (*R1)[j1].s - (*L2)[i2].s - (*R2)[j2].s;

			if (h_diff > 0)
			{
				if (j2 < S / 4 - 1)
				{
					heap2.pop();
					heap2.push(LRComb((*L2)[i2].h + (*R2)[j2 + 1].h, i2, j2 + 1));
				}
				else
				{
					heap2.pop();
				}
			}
			else
			{
				h_diff = -h_diff;
				s_diff = -s_diff;
				if (j1 < S / 4 - 1)
				{
					heap1.pop();
					heap1.push(LRComb((*L1)[i1].h + (*R1)[j1 + 1].h, i1, j1 + 1));
				}
				else
				{
					heap1.pop();
				}
			}

			if (h_diff < threshold)
			{
				sigsptr->emplace_back(SignatureSimple(h_diff, s_diff));
				num_col++;
				if (num_col % 100 == 0)
					printf("%u/%u collisions found, %.2lf %% done \n", num_col, counter, num_col * 100.0 / keep);
			}
		}
		if (num_col < keep)
			puts("WARNING: failed to get expected amount of reduced values!");
		std::cout << "completed after " << counter << " loops" << std::endl;
		puts("-------------------------");
		delete L1;
		delete R1;
		delete L2;
		delete R2;
	}
}

/* Sort-and-difference code */
void sort_and_difference(std::vector<SignatureSC25519> &sigs, const uint32_t n_bit, const uint32_t l, const uint32_t b,
						 const uint32_t filter, const uint32_t a, const int log_prec, const size_t iota = 1, const std::string out_prefix = "", const bool istest = true)
{
	uint32_t threshold_bit = n_bit - filter;
	uint32_t S;

	for (size_t round = 0; round < iota; round++)
	{
		S = sigs.size();
		threshold_bit -= b;
		mpz_class threshold_mpz = mpz_class(1) << threshold_bit;
		sc25519 threshold_sc;
		mpz_to_gs(threshold_sc, threshold_mpz);

		std::sort(sigs.begin(), sigs.end());

		sc25519 newh, news;
		uint32_t j = 0;
		for (uint32_t i = 0; i < S - 1; i++)
		{
			sc25519_sub(&newh, &(sigs[i + 1].h), &(sigs[i].h));
			if (sc25519_lt(&newh, &threshold_sc))
			{
				sc25519_sub(&news, &(sigs[i + 1].s), &(sigs[i].s));
				sigs[j] = SignatureSC25519(newh, news);
				j++;
			}
		}
		sigs.resize(j);
		printf("sorting done; %lu elements with h < 2^%u obtained\n", sigs.size(), threshold_bit);
		if (out_prefix.length())
		{
			// file format: redsigs_sd_round-i.bin
			std::string outsig = out_prefix + "_round-" + std::to_string(round) + ".bin";
			printf("saving signatures to %s... \n", outsig.c_str());
			sigsave(sigs, outsig);
		}
	}
	return;
}

void sort_and_difference(std::vector<SignatureSimple> &sigs, mpz_class &bound, const uint32_t n_bit, const uint32_t l,
						 const uint32_t b, const uint32_t filter, const uint32_t a, const int log_prec,
						 const size_t iota = 1, const std::string out_prefix = "", std::string dir = "", const bool istest = true)
{
	uint32_t threshold_bit = n_bit - filter;
	mpz_class threshold_mpz;
	uint32_t S;
	if (dir.length())
		dir += "/";

	for (size_t round = 0; round < iota; round++)
	{
		S = sigs.size();
		threshold_bit -= b;
		threshold_mpz = mpz_class(1) << threshold_bit;
		std::sort(sigs.begin(), sigs.end());

		mpz_class newh, news;
		uint32_t j = 0;
		for (uint32_t i = 0; i < S - 1; i++)
		{
			newh = sigs[i + 1].h - sigs[i].h;
			if (newh < threshold_mpz)
			{
				news = sigs[i + 1].s - sigs[i].s;
				sigs[j] = SignatureSimple(newh, news);
				j++;
			}
		}
		sigs.resize(j);
		printf("sorting done; %lu elements with h < 2^%u obtained\n", sigs.size(), threshold_bit);
		bound = threshold_mpz.get_d();
		if (out_prefix.length())
		{
			// file format: redsigs_sd_round-i.bin
			std::string outsig = out_prefix + "_round-" + std::to_string(round) + ".bin";
			printf("saving signatures to %s... \n", (dir + outsig).c_str());
			sigsave(sigs, dir + outsig);
		}
	}
	return;
}
