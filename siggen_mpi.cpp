//============================================================================
// Name        : siggen_mpi.cpp
// Author      : Akira Takahashi and Mehdi Tibouchi
// Version     : 0.1
// Copyright   : Public domain
// Description : Faulty qDSA signature generator
// Standard    : C++11
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <inttypes.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <bitset>

#include <gmpxx.h>
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <boost/program_options.hpp>
#include <random>

#include "mocksig.h"

namespace mpi = boost::mpi;
namespace po = boost::program_options;

int main(int argc, char **argv)
{
    /* MPI */
    mpi::environment env;
    mpi::communicator world;
    const int master = 0;

    /* parse command line arguments */
    po::options_description desc("Allowed options");
    desc.add_options()("help", "help message")("verbose", "verbose logging")("test10", "generate test signatures 10-bit")("test60", "generate test signatures 60-bit")("test90", "generate test signatures 90-bit")("test131", "generate test signatures 131-bit")("prime163r1", "generate test signatures 162-bit")("prime192v1", "generate test signatures 192-bit")("out", po::value<std::string>(), "save signature data to a file with specified prefix")("leak", po::value<int>(), "number of nonce LSBs to be leaked")("filter", po::value<int>(), "number of bits filtered")("error-rate", po::value<float>(), "number of error rate")("msbs", po::value<int>(), "MSBs value")("uniform", "generate unirorm biased nonces")

        ;

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
        log_prec *= 10;
    }

    /* Parameters */
    uint32_t n_bit;      // 鍵長
    mpz_class n, tmplim; //
    sc25519 lim;         //
    uint32_t a;          //
    uint32_t filter;
    uint32_t leak;                          // 漏洩しているnonceの長さ
    uint32_t S;                             // 作成する署名数
    Domain pp;                              // ドメイン
    mpz_class d;                            // 秘密鍵
    std::vector<SignatureSimple> subresult; // 署名組のvector
    float epsilon;                          // error rate
    bool consider_error = false;

    if (vm.count("leak"))
        leak = vm["leak"].as<int>();
    else
        leak = 0;

    std::cout << "leak: " << leak << std::endl;

    if (vm.count("filter"))
        filter = vm["filter"].as<int>();
    else
        filter = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    if (vm.count("error-rate"))
    {
        epsilon = vm["error-rate"].as<float>();
        if (epsilon != 0)
        {
            consider_error = true;
        }
    }
    else
    {
        epsilon = 0;
    }

    if (vm.count("test10"))
    {
        /* Use mock signature */
        n_bit = 10;
        n = (mpz_class(1) << n_bit) - 27;
        tmplim = n >> filter;
        a = 8;
        S = 1 << (a + 2);
        pp = mock::setup(n_bit, n);
        d = mpz_class("829");
        gmp_printf("pp = (n_bit=%lu, n=%Zd), d=%Zd\n", pp.n_bit, pp.n.get_mpz_t(), d.get_mpz_t());
        gmp_randclass rand(gmp_randinit_default);
        rand.seed(1234567);

        if (world.rank() == master)
        {
            printf("master: generating %u signatures with |h| <= %u bits...\n", S, n_bit - filter);
            uint32_t numsigs = 0;
            mpz_class div, inv, h_tmp, s_tmp;
            div = mpz_class(1) << leak;
            mpz_class msbs = mpz_class(5);
            if (vm.count("msbs"))
            {
                msbs = mpz_class(vm["msbs"].as<int>());
            }
            std::vector<int> msb_vector = mock::calc_msb_vector(msbs, leak);
            bool incorrect_flag = false;
            std::vector<int> error_vector;
            while (subresult.size() < S)
            {
                numsigs++;
                // SignatureLeak sig = mock::sign_msb_leak(pp, d, 0, leak, rand);
                // std::cout <<"generation: " << numsigs << std::endl;
                // SignatureLeak sig = mock::sign_msbs_leak(pp, d, 0, leak, rand, mpz_class(5));

                if (consider_error)
                {
                    error_vector = mock::sign_error_vector(leak, msb_vector, dis, gen, epsilon);
                }

                SignatureLeak sig = consider_error ? mock::sign_msbs_leak_error(pp, d, 0, leak, rand, error_vector) : mock::sign_msbs_leak(pp, d, 0, leak, rand, mpz_class(5));

                // std::cout <<"generated: " << numsigs << std::endl;
                // h_tmp = sig.h*inv;
                // mpz_mod(sig.h.get_mpz_t(), h_tmp.get_mpz_t(), pp.n.get_mpz_t());
                // if (sig.h >= tmplim) continue;
                // s_tmp = (sig.s - sig.rr)*inv;
                // mpz_mod(sig.s.get_mpz_t(), s_tmp.get_mpz_t(), pp.n.get_mpz_t());
                subresult.emplace_back(SignatureSimple(sig.h, sig.s));
                // std::cout <<"S/log_prec"<< (S/log_prec) << std::endl;

                // std::cout <<"pushed: " << numsigs << std::endl;
                // if (subresult.size() % (S/log_prec) == 0) printf("%.2f %% done\n", (float)subresult.size()*100/S);
            }
            printf("master: got %u/%u signatures\n", S, numsigs);
        }
    }
    else if (vm.count("test60"))
    {
        /* Use mock signature */
        n_bit = 60;
        n = (mpz_class(1) << n_bit) - 1061862795;
        tmplim = n >> filter;
        a = 6;
        S = 1 << (a + 2);
        pp = mock::setup(n_bit, n);
        d = mpz_class("302038621189435203");
        gmp_printf("pp = (n_bit=%lu, n=%Zd), d=%Zd\n", pp.n_bit, pp.n.get_mpz_t(), d.get_mpz_t());
        gmp_randclass rand(gmp_randinit_default);
        rand.seed(1234567);

        if (world.rank() == master)
        {
            printf("master: generating %u signatures with |h| <= %u bits...\n", S, n_bit - filter);
            uint32_t numsigs = 0;
            mpz_class div, inv, h_tmp, s_tmp;
            div = mpz_class(1) << leak;
            mpz_invert(inv.get_mpz_t(), div.get_mpz_t(), pp.n.get_mpz_t());
            uint32_t count = 0;
            mpz_class msbs = mpz_class(1);
            if (vm.count("msbs"))
            {
                msbs = mpz_class(vm["msbs"].as<int>());
            }
            std::vector<int> msb_vector = mock::calc_msb_vector(msbs, leak);
            std::vector<int> error_vector;
            while (subresult.size() < S / 4)
            {
                numsigs++;
                // SignatureLeak sig = mock::sign_msb_leak(pp, d, 0, leak, rand);
                if (consider_error)
                {
                    error_vector = mock::sign_error_vector(leak, msb_vector, dis, gen, epsilon);
                }

                SignatureLeak sig = consider_error ? mock::sign_msbs_leak_error(pp, d, 0, leak, rand, error_vector) : mock::sign_msbs_leak(pp, d, 0, leak, rand, msbs);

                // h_tmp = sig.h*inv;
                // mpz_mod(sig.h.get_mpz_t(), h_tmp.get_mpz_t(), pp.n.get_mpz_t());
                // if (sig.h >= tmplim) continue;
                // s_tmp = (sig.s - sig.rr)*inv;
                // mpz_mod(sig.s.get_mpz_t(), s_tmp.get_mpz_t(), pp.n.get_mpz_t());
                subresult.emplace_back(SignatureSimple(sig.h, sig.s));
                if (subresult.size() % (S / log_prec) == 0)
                    printf("%.2f %% done\n", (float)subresult.size() * 100 / S);
            }
            std::cout << "change 1" << std::endl;
            while (subresult.size() < S / 2)
            {
                numsigs++;
                // SignatureLeak sig = mock::sign_msb_leak(pp, d, 0, leak, rand);
                if (consider_error)
                {
                    error_vector = mock::sign_error_vector(leak, msb_vector, dis, gen, epsilon);
                }

                SignatureLeak sig = consider_error ? mock::sign_msbs_leak_error(pp, d, 0, leak, rand, error_vector) : mock::sign_msbs_leak(pp, d, 0, leak, rand, mpz_class(0));

                // h_tmp = sig.h*inv;
                // mpz_mod(sig.h.get_mpz_t(), h_tmp.get_mpz_t(), pp.n.get_mpz_t());
                // if (sig.h >= tmplim) continue;
                // s_tmp = (sig.s - sig.rr)*inv;
                // mpz_mod(sig.s.get_mpz_t(), s_tmp.get_mpz_t(), pp.n.get_mpz_t());
                subresult.emplace_back(SignatureSimple(sig.h, sig.s));
                if (subresult.size() % (S / log_prec) == 0)
                    printf("%.2f %% done\n", (float)subresult.size() * 100 / S);
            }

            std::cout << "change 2" << std::endl;
            while (subresult.size() < 3 * S / 4)
            {
                numsigs++;
                // SignatureLeak sig = mock::sign_msb_leak(pp, d, 0, leak, rand);
                if (consider_error)
                {
                    error_vector = mock::sign_error_vector(leak, msb_vector, dis, gen, epsilon);
                }

                SignatureLeak sig = consider_error ? mock::sign_msbs_leak_error(pp, d, 0, leak, rand, error_vector) : mock::sign_msbs_leak(pp, d, 0, leak, rand, mpz_class(2));

                // h_tmp = sig.h*inv;
                // mpz_mod(sig.h.get_mpz_t(), h_tmp.get_mpz_t(), pp.n.get_mpz_t());
                // if (sig.h >= tmplim) continue;
                // s_tmp = (sig.s - sig.rr)*inv;
                // mpz_mod(sig.s.get_mpz_t(), s_tmp.get_mpz_t(), pp.n.get_mpz_t());
                subresult.emplace_back(SignatureSimple(sig.h, sig.s));
                if (subresult.size() % (S / log_prec) == 0)
                    printf("%.2f %% done\n", (float)subresult.size() * 100 / S);
            }

            std::cout << "change 3" << std::endl;
            while (subresult.size() < S)
            {
                numsigs++;
                // SignatureLeak sig = mock::sign_msb_leak(pp, d, 0, leak, rand);
                if (consider_error)
                {
                    error_vector = mock::sign_error_vector(leak, msb_vector, dis, gen, epsilon);
                }

                SignatureLeak sig = consider_error ? mock::sign_msbs_leak_error(pp, d, 0, leak, rand, error_vector) : mock::sign_msbs_leak(pp, d, 0, leak, rand, mpz_class(3));

                // h_tmp = sig.h*inv;
                // mpz_mod(sig.h.get_mpz_t(), h_tmp.get_mpz_t(), pp.n.get_mpz_t());
                // if (sig.h >= tmplim) continue;
                // s_tmp = (sig.s - sig.rr)*inv;
                // mpz_mod(sig.s.get_mpz_t(), s_tmp.get_mpz_t(), pp.n.get_mpz_t());
                subresult.emplace_back(SignatureSimple(sig.h, sig.s));
                if (subresult.size() % (S / log_prec) == 0)
                    printf("%.2f %% done\n", (float)subresult.size() * 100 / S);
            }
            printf("master: got %u/%u signatures\n", S, numsigs);
        }
    }
    else if (vm.count("test90"))
    {
        /* Use mock signature */
        n_bit = 90;
        n = (mpz_class(1) << n_bit) - 33;
        tmplim = n >> filter;
        a = 14;
        S = 1 << (a + 2);
        pp = mock::setup(n_bit, n);
        d = mpz_class("924408261565060156037890712");
        gmp_printf("pp = (n_bit=%lu, n=%Zd), d=%Zd\n", pp.n_bit, pp.n.get_mpz_t(), d.get_mpz_t());
        gmp_randclass rand(gmp_randinit_default);
        rand.seed(1234567);

        if (world.rank() == master)
        {
            printf("master: generating %u signatures with |h| <= %u bits...\n", S, n_bit - filter);
            uint32_t numsigs = 0;
            mpz_class div, inv, h_tmp, s_tmp;
            div = mpz_class(1) << leak;
            mpz_invert(inv.get_mpz_t(), div.get_mpz_t(), pp.n.get_mpz_t());
            bool incorrect_flag = false;
            mpz_class msbs = mpz_class(3);
            if (vm.count("msbs"))
            {
                msbs = mpz_class(vm["msbs"].as<int>());
            }
            std::vector<int> msb_vector = mock::calc_msb_vector(msbs, leak);
            std::vector<int> error_vector;
            while (subresult.size() < S)
            {
                numsigs++;
                // SignatureLeak sig = mock::sign_msb_leak(pp, d, 0, leak, rand);
                // SignatureLeak sig = mock::sign_msbs_leak(pp, d, 0, leak, rand, mpz_class(5));

                if (consider_error)
                {
                    error_vector = mock::sign_error_vector(leak, msb_vector, dis, gen, epsilon);
                }

                SignatureLeak sig = consider_error ? mock::sign_msbs_leak_error(pp, d, 0, leak, rand, error_vector) : mock::sign_msbs_leak(pp, d, 0, leak, rand, msbs);

                // h_tmp = sig.h*inv;
                // mpz_mod(sig.h.get_mpz_t(), h_tmp.get_mpz_t(), pp.n.get_mpz_t());
                // if (sig.h >= tmplim) continue;
                // s_tmp = (sig.s - sig.rr)*inv;
                // mpz_mod(sig.s.get_mpz_t(), s_tmp.get_mpz_t(), pp.n.get_mpz_t());
                subresult.emplace_back(SignatureSimple(sig.h, sig.s));
                if (subresult.size() % (S / log_prec) == 0)
                    printf("%.2f %% done\n", (float)subresult.size() * 100 / S);
            }
            printf("master: got %u/%u signatures\n", S, numsigs);
        }
    }
    else if (vm.count("prime192v1"))
    {
        /* Use mock signature */
        n_bit = 192;
        n = (mpz_class(1) << n_bit) - mpz_class("31607402335160671281192228815");
        tmplim = n >> filter;
        a = 22;
        S = 1 << (a + 2);
        pp = mock::setup(n_bit, n);
        d = mpz_class("4336966521141612760869415195855092141770523415923");
        gmp_printf("pp = (n_bit=%lu, n=%Zd), d=%Zd\n", pp.n_bit, pp.n.get_mpz_t(), d.get_mpz_t());
        gmp_randclass rand(gmp_randinit_default);
        rand.seed(1234567);

        if (world.rank() == master)
        {
            printf("master: generating %u signatures with |h| <= %u bits...\n", S, n_bit - filter);
            uint32_t numsigs = 0;
            mpz_class div, inv, h_tmp, s_tmp;
            div = mpz_class(1) << leak;
            mpz_invert(inv.get_mpz_t(), div.get_mpz_t(), pp.n.get_mpz_t());
            bool incorrect_flag = false;
            mpz_class msbs = mpz_class(5);
            if (vm.count("msbs"))
            {
                msbs = mpz_class(vm["msbs"].as<int>());
            }
            std::vector<int> msb_vector = mock::calc_msb_vector(msbs, leak);
            std::vector<int> error_vector;
            while (subresult.size() < S)
            {
                numsigs++;
                // SignatureLeak sig = mock::sign_msb_leak(pp, d, 0, leak, rand);
                if (consider_error)
                {
                    error_vector = mock::sign_error_vector(leak, msb_vector, dis, gen, epsilon);
                }
                SignatureLeak sig = consider_error ? mock::sign_msbs_leak_error(pp, d, 0, leak, rand, error_vector) : mock::sign_msbs_leak(pp, d, 0, leak, rand, mpz_class(5));

                // h_tmp = sig.h*inv;
                // mpz_mod(sig.h.get_mpz_t(), h_tmp.get_mpz_t(), pp.n.get_mpz_t());
                // if (sig.h >= tmplim) continue;
                // s_tmp = (sig.s - sig.rr)*inv;
                // mpz_mod(sig.s.get_mpz_t(), s_tmp.get_mpz_t(), pp.n.get_mpz_t());
                subresult.emplace_back(SignatureSimple(sig.h, sig.s));
                if (subresult.size() % (S / log_prec) == 0)
                    printf("%.2f %% done\n", (float)subresult.size() * 100 / S);
            }
            printf("master: got %u/%u signatures\n", S, numsigs);
        }
    }
    else if (vm.count("prime163r1"))
    {
        /* Use mock signature */
        n_bit = 162;
        n = (mpz_class(1) << n_bit) - mpz_class("865766333097319309760613");
        tmplim = n >> filter;
        a = 23;
        S = 1 << (a + 2);
        pp = mock::setup(n_bit, n);
        d = mpz_class("4336966521141612760869415195855092141770523415923");
        gmp_printf("pp = (n_bit=%lu, n=%Zd), d=%Zd\n", pp.n_bit, pp.n.get_mpz_t(), d.get_mpz_t());
        gmp_randclass rand(gmp_randinit_default);
        rand.seed(1234567);

        if (world.rank() == master)
        {
            printf("master: generating %u signatures with |h| <= %u bits...\n", S, n_bit - filter);
            uint32_t numsigs = 0;
            mpz_class div, inv, h_tmp, s_tmp;
            div = mpz_class(1) << leak;
            mpz_invert(inv.get_mpz_t(), div.get_mpz_t(), pp.n.get_mpz_t());
            bool incorrect_flag = false;
            mpz_class msbs = mpz_class(1);
            if (vm.count("msbs"))
            {
                msbs = mpz_class(vm["msbs"].as<int>());
            }
            int count_error = 0;
            std::vector<int> msb_vector = mock::calc_msb_vector(msbs, leak);
            std::vector<int> error_vector;
            while (subresult.size() < S)
            {
                numsigs++;
                // SignatureLeak sig = mock::sign_msbs_leak(pp, d, 0, leak, rand, mpz_class(5));if (consider_error)
                if (consider_error)
                {
                    error_vector = mock::sign_error_vector(leak, msb_vector, dis, gen, epsilon);
                }
                SignatureLeak sig = consider_error ? mock::sign_msbs_leak_error(pp, d, 0, leak, rand, error_vector) : mock::sign_msbs_leak(pp, d, 0, leak, rand, msbs);

                // h_tmp = sig.h*inv;
                // mpz_mod(sig.h.get_mpz_t(), h_tmp.get_mpz_t(), pp.n.get_mpz_t());
                // if (sig.h >= tmplim) continue;
                // s_tmp = (sig.s - sig.rr)*inv;
                // mpz_mod(sig.s.get_mpz_t(), s_tmp.get_mpz_t(), pp.n.get_mpz_t());
                if (sig.rr != msbs)
                {
                    count_error++;
                }
                subresult.emplace_back(SignatureSimple(sig.h, sig.s));

                if (numsigs % 1000000 == 0)
                {
                    std::cout << numsigs << " signatures has made." << std::endl;
                }

                if (subresult.size() % (S / log_prec) == 0)
                    printf("%.2f %% done\n", (float)subresult.size() * 100 / S);
            }
            std::cout << "error/total: " << ((double)count_error / numsigs) << std::endl;
            printf("master: got %u/%u signatures\n", S, numsigs);
        }
    }
    else if (vm.count("test131"))
    {
        /* Use mock signature */
        n_bit = 131;
        n = (mpz_class(1) << n_bit) - 681;
        tmplim = n >> filter;
        a = 21;
        // a = 26;
        S = 1 << (a + 2);
        pp = mock::setup(n_bit, n);
        d = mpz_class("2690717704635825153895385645374096400717");
        // d = mpz_class("1647779926160126558973906702628759962437");
        // d = mpz_class("1361129467683753853853498429727072845483");
        gmp_printf("pp = (n_bit=%lu, n=%Zd), d=%Zd\n", pp.n_bit, pp.n.get_mpz_t(), d.get_mpz_t());
        gmp_randclass rand(gmp_randinit_default);
        rand.seed(1234567);

        if (world.rank() == master)
        {
            printf("master: generating %u signatures with |h| <= %u bits...\n", S, n_bit - filter);
            uint32_t numsigs = 0;
            mpz_class div, inv, h_tmp, s_tmp;
            div = mpz_class(1) << leak;
            mpz_invert(inv.get_mpz_t(), div.get_mpz_t(), pp.n.get_mpz_t());
            mpz_class msbs = mpz_class(1);

            int count_error = 0;
            std::vector<int> error_vector;

            std::cout << "consider_error: " << consider_error << std::endl;

            int msb_num = std::pow(2, leak);
            if (vm.count("uniform"))
            {

                for (int msbs_num = 0; msbs_num < msb_num; msbs_num++)
                {
                    while (subresult.size() < (msbs_num + 1) * S / msb_num)
                    {
                        numsigs++;

                        msbs = mpz_class(msbs_num);
                        std::vector<int> msb_vector = mock::calc_msb_vector(msbs, leak);
                        if (consider_error)
                        {
                            error_vector = mock::sign_error_vector(leak, msb_vector, dis, gen, epsilon);
                        }

                        SignatureLeak sig = consider_error ? mock::sign_msbs_leak_error(pp, d, 0, leak, rand, error_vector) : mock::sign_msbs_leak(pp, d, 0, leak, rand, msbs);
                        if (sig.rr != msbs)
                        {
                            count_error++;
                        }
                        // h_tmp = sig.h*inv;
                        // mpz_mod(sig.h.get_mpz_t(), h_tmp.get_mpz_t(), pp.n.get_mpz_t());
                        // if (sig.h >= tmplim) continue;
                        // s_tmp = (sig.s - sig.rr)*inv;
                        // mpz_mod(sig.s.get_mpz_t(), s_tmp.get_mpz_t(), pp.n.get_mpz_t());
                        subresult.emplace_back(SignatureSimple(sig.h, sig.s));
                        if (subresult.size() % (S / log_prec) == 0)
                            printf("%.2f %% done\n", (float)subresult.size() * 100 / S);
                    }
                    std::cout << "msbs_num: " << msbs_num << ", numsigs: " << numsigs << std::endl;
                }
            }
            else
            {
                if (vm.count("msbs"))
                {
                    msbs = mpz_class(vm["msbs"].as<int>());
                }
                const std::vector<int> msb_vector = mock::calc_msb_vector(msbs, leak);
                while (subresult.size() < S)
                {
                    numsigs++;
                    // SignatureLeak sig = mock::sign(pp, d, 0, leak, rand);
                    // SignatureLeak sig = mock::sign_msbs_leak(pp, d, 0, leak, rand, mpz_class(5));

                    if (consider_error)
                    {
                        error_vector = mock::sign_error_vector(leak, msb_vector, dis, gen, epsilon);
                    }
                    SignatureLeak sig = consider_error ? mock::sign_msbs_leak_error(pp, d, 0, leak, rand, error_vector) : mock::sign_msbs_leak(pp, d, 0, leak, rand, msbs);

                    if (sig.rr != msbs)
                    {
                        count_error++;
                    }
                    // h_tmp = sig.h*inv;
                    // mpz_mod(sig.h.get_mpz_t(), h_tmp.get_mpz_t(), pp.n.get_mpz_t());
                    // if (sig.h >= tmplim) continue;
                    // s_tmp = (sig.s - sig.rr)*inv;
                    // mpz_mod(sig.s.get_mpz_t(), s_tmp.get_mpz_t(), pp.n.get_mpz_t());
                    if (numsigs % 1000000 == 0)
                    {
                        std::cout << numsigs << " signatures has made." << std::endl;
                        std::cout << count_error << " signatures have errors" << std::endl;
                    }
                    subresult.emplace_back(SignatureSimple(sig.h, sig.s));
                    if (subresult.size() % (S / log_prec) == 0)
                        printf("%.2f %% done\n", (float)subresult.size() * 100 / S);
                }
            }
            std::cout << "error/total: " << ((double)count_error / numsigs) << std::endl;
            printf("master: got %u/%u signatures\n", S, numsigs);
        }
    }
    else
    {
        /* Use real qDSA */
        n_bit = 252;
        n = (mpz_class(1) << 252) + mpz_class("27742317777372353535851937790883648493");
        tmplim = n >> filter;
        mpz_to_gs(lim, tmplim);
        a = 24;
        S = 1 << (a + 2);
        pp = mock::setup(n_bit, n);
        const unsigned long long mlen = sizeof(uint64_t) + sizeof(int); // should be > (a+2+filter)/8
        size_t S_sub;
        if (world.rank() == master)
        {
            S_sub = (S + world.size() - 1) / world.size();
            printf("master : generating %u signatures...\n", S);
        }
        broadcast(world, S_sub, master);
        const size_t log_mod = S_sub / log_prec;
        const float percent_delta = 100.0 / log_prec;
        size_t log_th = 0;
        float percent = 0;
        if (leak != 0 && leak != 2 && leak != 3)
        {
            printf("%u-bit leak is not supported\n", leak);
            return 1;
        }

        /* KeyGen */
        uint8_t pk[32];
        uint8_t sk[64] = {0x8D, 0x1E, 0xBD, 0x7D, 0xF8, 0xAE, 0xF4, 0x53, 0x02, 0x49, 0xD3, 0x26, 0x17, 0x58, 0x7A, 0xAC,
                          0x44, 0x09, 0xA7, 0x79, 0x34, 0x5F, 0x1E, 0x6B, 0x20, 0x1C, 0xFC, 0x7C, 0xC5, 0x7E, 0x3C, 0x53,
                          0x9D, 0xF9, 0xD0, 0x95, 0xA7, 0xC4, 0xE9, 0xA9, 0x0D, 0xBC, 0xD0, 0x24, 0x14, 0x4A, 0xD0, 0x58,
                          0x54, 0x78, 0xD1, 0x88, 0xD7, 0xF0, 0xF4, 0xF7, 0x0C, 0xF0, 0x73, 0xD2, 0x6E, 0xAF, 0x25, 0x0B};
        // This leads to d = 5220582922658643192668885191618908575833980181104027493552863441828733052420
        gmp_randclass rand(gmp_randinit_default);
        rand.seed(1234567);
        d = mock::keygen(pp, rand);
        printf("d is %lu-bit\n", mpz_sizeinbase(d.get_mpz_t(), 2));
        mpz_mod(d.get_mpz_t(), d.get_mpz_t(), n.get_mpz_t());
        gmp_printf("pp=(n_bit=%u, n=%Zd),\n        d=%Zd\n", pp.n_bit, pp.n.get_mpz_t(), d.get_mpz_t());

        /* SigGen */
        printf("[%d]/[%d] : generating %lu signatures with |h| <= %u bits...\n", world.rank(), world.size(), S_sub, n_bit - filter);
        uint64_t numsigs = 0;

        union
        {
            unsigned char m[mlen]; // message
            struct
            {
                uint64_t index;
                int rank;
            } mstruct;
        };
        unsigned char sm[64 + mlen]; // +-R || s || m
        unsigned long long smlen;

        mstruct.rank = world.rank();
        while (subresult.size() < S_sub)
        {
            numsigs++;
            /* set message */
            mstruct.index = numsigs;
#if 0
			if(numsigs==17) {
			printf("[%d]/[%d] : m[%u] = ", world.rank(), world.size(), numsigs);
			for(unsigned long long i=0;i<mlen;i++)
				printf("%02x%c", m[i], (i==mlen-1)?'\n':':');
			}
#endif

            /* sign */
            SignatureSimple sig;
            if (leak == 0)
            {
                mock::sign(pp, d, 0, leak, rand);
            }
            else
            {
                // if (mock::sign_fault(sig, sm, &smlen, m, mlen, pk, sk, pp.n, &lim, leak) != 1) continue;
            }
            subresult.emplace_back(sig);
            if (subresult.size() >= log_th)
            {
                printf("[%d]/[%d]: %.2f %% done\n", world.rank(), world.size(), percent);
                log_th += log_mod;
                percent += percent_delta;
            }

            /* verify signature */
#if 0
            int ch = mock::verify(m, mlen, sm, smlen, pk);
            if (ch != 1) printf("WARNING: invalid signature\n");
#endif
        }
        printf("[%d]/[%d]: got %lu/%lu signatures\n", world.rank(), world.size(), S_sub, numsigs);
    }

    if (vm.count("out"))
    {
        std::string outsig = vm["out"].as<std::string>() + "_" + std::to_string(world.rank()) + ".bin";
        printf("master: saving %lu signatures in %s... \n", subresult.size(), outsig.c_str());
        sigsave(subresult, outsig);
    }
    return EXIT_SUCCESS;
}
