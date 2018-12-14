#include "PrimeOmp.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

PrimeOmp::PrimeOmp(number_t a, number_t b)
		:	Prime(a, b) {

}

int PrimeOmp::Find() {
	omp_set_num_threads(NUM_THREADS);
	std::vector<std::list<number_t>> lists(NUM_THREADS);

	#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		#pragma omp for schedule(guided)
		for (number_t n = border_a_; n <= border_b_; ++n)
			if (Check(n))
				lists[tid].push_back(n);
	}

	for (auto &list : lists)
		primes_list.merge(list);

	found_ = primes_list.size();

	return 0;
}
