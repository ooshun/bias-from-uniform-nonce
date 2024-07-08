#ifndef REDUCTION_H
#define REDUCTION_H

class LRComb
{

public:
	mpz_class hsum;
	uint32_t idx_L;
	uint32_t idx_R;
	LRComb(mpz_class, uint32_t, uint32_t);
	bool operator<(const LRComb &c) const
	{
		return (hsum > c.hsum);
	}
	bool operator>(const LRComb &c) const
	{
		return (hsum < c.hsum);
	}
};

class LRCombGSS
{

public:
	gss hsum;
	uint32_t idx_L;
	uint32_t idx_R;
	LRCombGSS() : hsum({.v = {0, 0}}), idx_L(0), idx_R(0)
	{
	}
	LRCombGSS(gss hh, uint32_t i, uint32_t j) : hsum(hh), idx_L(i), idx_R(j)
	{
	}
	bool operator<(const LRCombGSS &c) const
	{
		return (gss_lt(&hsum, &(c.hsum)) == 1);
	}
	bool operator>(const LRCombGSS &c) const
	{
		return (gss_lt(&hsum, &(c.hsum)) == 0);
	}
};

#if 1
class LRComb64
{

public:
	uint64_t hsum;
	uint32_t idx_L;
	uint32_t idx_R;

	LRComb64(uint64_t hh = 0, uint32_t i = 0, uint32_t j = 0) : hsum(hh), idx_L(i), idx_R(j)
	{
	}
	inline bool operator<(const LRComb64 &c) const
	{
		return hsum < c.hsum;
	}
	inline bool operator>(const LRComb64 &c) const
	{
		return hsum > c.hsum;
	}
	uint64_t operator>>(const unsigned offset)
	{
		return hsum >> offset;
	}
};
#endif

#if 1
class LRComb128
{

public:
	uint128_t hsum;
	uint32_t idx_L;
	uint32_t idx_R;
	LRComb128() : hsum(0), idx_L(0), idx_R(0)
	{
	}
	LRComb128(uint128_t hh, uint32_t i, uint32_t j) : hsum(hh), idx_L(i), idx_R(j)
	{
	}
	inline bool operator<(const LRComb128 &c) const
	{
		return hsum < c.hsum;
	}
	inline bool operator>(const LRComb128 &c) const
	{
		return hsum > c.hsum;
	}
	uint128_t operator>>(const unsigned offset)
	{
		return hsum >> offset;
	}
};
#elif 0
class LRComb96
{

public:
	uint96_t hsum;
	uint32_t idx_L;
	uint32_t idx_R;
	LRCombUint() : hsum(0), idx_L(0), idx_R(0)
	{
	}
	LRCombUint(uint96_t hh, uint32_t i, uint32_t j) : hsum(hh), idx_L(i), idx_R(j)
	{
	}
	inline bool operator<(const LRCombUint &c) const
	{
		return hsum < c.hsum;
	}
	inline bool operator>(const LRCombUint &c) const
	{
		return hsum > c.hsum;
	}
	uint96_t operator>>(const unsigned offset)
	{
		return hsum >> offset;
	}
};
#endif

class Index
{
private:
	friend class boost::serialization::access;

	template <class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar & idx_L1;
		ar & idx_R1;
		ar & idx_L2;
		ar & idx_R2;
		ar & flip;
	}

public:
	uint32_t idx_L1;
	uint32_t idx_R1;
	uint32_t idx_L2;
	uint32_t idx_R2;
	bool flip;
	Index() : idx_L1(0), idx_R1(0), idx_L2(0), idx_R2(0), flip(false){};
	Index(uint32_t i1, uint32_t j1, uint32_t i2, uint32_t j2, bool f = false) : idx_L1(i1), idx_R1(j1), idx_L2(i2), idx_R2(j2), flip(f)
	{
	}
};

void idxsave(std::vector<Index> &is, const std::string &filename);
void idxload(std::vector<Index> &is, const std::string &filename);
void restore_from_idx(std::vector<SignatureSimple> &sigs, const std::vector<Index> &idxlist, const std::vector<SignatureSimple> &L1,
					  const std::vector<SignatureSimple> &R1, const std::vector<SignatureSimple> &L2, const std::vector<SignatureSimple> &R2,
					  const mpz_class &threshold_mpz);

void schroeppel_shamir(std::vector<SignatureSimple> *sigs, const uint32_t n_bit, const uint32_t l, const uint32_t b, const uint32_t filter,
					   const size_t iota);

void schroeppel_shamir_mpi(std::vector<SignatureSimple> &sigs, mpz_class &bound, const uint32_t n_bit, const uint32_t l, const uint32_t b,
						   const uint32_t filter, uint32_t a, const int log_prec, const std::vector<uint32_t> &b_info,
						   const std::vector<uint32_t> &l_info, const size_t iota, const std::string out_prefix, const bool out_index, std::string dir, const bool istest);

void sort_and_difference(std::vector<SignatureSimple> &sigs, mpz_class &bound, const uint32_t n_bit, const uint32_t l, const uint32_t b,
						 const uint32_t filter, const uint32_t a, const int log_prec, const size_t iota, const std::string out_prefix, std::string dir,
						 const bool istest);

void sort_and_difference(std::vector<SignatureSC25519> &sigs, const uint32_t n_bit, const uint32_t l, const uint32_t b, const uint32_t filter,
						 const uint32_t a, const int log_prec, const size_t iota, const std::string out_prefix, const bool istest);

void exhaustive_four_sum(std::vector<SignatureSimple> &sigs, const uint32_t threshold_bit, const uint32_t keep_max, const int log_prec);

/// @brief function of 4-list sum algorithm
/// @param sigs signatures
/// @param threshold_bit v
/// @param ignore_bit n
/// @param a size of list
/// @param m_double logarithm of the number of outputs
/// @param l_bit bit length
/// @param keep_max M'
/// @param round current round
/// @param dir output directory path
void parametarized_four_list_sum(std::vector<SignatureSimple> &sigs, const uint32_t threshold_bit, const uint32_t ignore_bit, const uint32_t a, const double m_double, const double mp, const uint32_t l_bit, const uint32_t keep_max, const int round, std::string dir, std::string out_prefix);

/**
 * @brief fuction to pass to 4-list sum algorith
 * @param sigs signatures
 * @param threshold_bit_vec Vector of v for r rounds
 * @param ignore_bit_vec Vector of n for r rounds
 * @param a_vec Vector of a for r rounds
 * @param m_vec Vector of m for r rounds
 * @param l_bit Remaining non-zero bit length
 * @param keep_max M'
 * @param out_prefix
 * @param dir
 */
void iterative_HGJ_four_list_sum(std::vector<SignatureSimple> &sigs, const std::vector<uint32_t> threshold_bit_vec, const std::vector<uint32_t> ignore_bit_vec, const std::vector<uint32_t> a_vec, const std::vector<double> m_vec, const uint32_t l_bit, const uint32_t keep_max, std::string out_prefix, std::string dir);

void calc_add_MSB_pair(std::vector<uint128_t> &L1_h, std::vector<uint128_t> &L2_h, const uint128_t &c, const uint32_t &l_bit, const uint32_t a_bit, const int &ofst, const int &pad, std::vector<LRComb64> &outputList);

uint32_t lower_bound_serach(std::vector<LRComb128> &combs, const uint32_t &lower_index, const uint32_t &upper_index, const uint128_t &hj_cand);

#endif

uint64_t calc_part_sum(const uint32_t &start, const uint32_t &end);

uint64_t calc_sum_square(const uint32_t &n);

uint64_t calc_part_sum_square(const uint32_t &start, const uint32_t &end);

uint64_t calc_numLnP(const uint32_t &a, const uint32_t &c);

uint64_t calc_numLP(const uint32_t &a, const uint32_t &c, const uint32_t &n);

uint64_t calc_num_out(const uint32_t &a, const uint32_t &v, const uint32_t &n);