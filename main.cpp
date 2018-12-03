#include <cstdlib>
#include <cstdio>
#include <chrono>

#include "PrimePosix.h"
#include "PrimeCpp11.h"
#include "PrimeOmp.h"

typedef std::chrono::steady_clock sc_t;
typedef sc_t::time_point tp_t;

int main(int argc, char** argv) {
	if (argc < 3)
		return 1;

	number_t a = strtoul(argv[1], nullptr, 0);
	number_t b = strtoul(argv[2], nullptr, 0);

	fprintf(stderr, "Single:\n");
	tp_t begin = sc_t::now();
	Prime finder(a, b);

	finder.Find();
	finder.Print();

	fprintf(stderr, "Posix:\n");
	tp_t inter = sc_t::now();
	PrimePosix finder_posix(a, b);

	finder_posix.Find();
	finder_posix.Print();

	fprintf(stderr, "CPP11:\n");
	tp_t end1 = sc_t::now();
	PrimeCpp11 finder_cpp11(a, b);

	finder_cpp11.Find();
	finder_cpp11.Print();

	fprintf(stderr, "OpenMP:\n");
	tp_t end2 = sc_t::now();
	PrimeOmp finder_omp(a, b);

	finder_omp.Find();
	finder_omp.Print();

	tp_t end3 = sc_t::now();

	double time_single = std::chrono::duration_cast<std::chrono::milliseconds>(inter - begin).count();
	double time_posix = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - inter).count();
	double time_cpp11 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end1).count();
	double time_omp = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - end2).count();

	printf("Single:\t%lf s\n", time_single / 1000);
	printf("POSIX:\t%lf s\n", time_posix / 1000);
	printf("CPP11:\t%lf s\n", time_cpp11 / 1000);
	printf("OpenMP:\t%lf s\n", time_omp / 1000);

	return 0;
}