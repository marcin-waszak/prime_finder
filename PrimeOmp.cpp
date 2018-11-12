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
	std::atomic_ullong current(border_a_);

	#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		while(1) {
			number_t n = current.fetch_add(1);

			if (n > border_b_)
				break;

			if (Check(n))
				lists[tid].push_back(n);
		}
	}

	for (auto &list : lists)
		primes_list.merge(list);

	found_ = primes_list.size();

	return 0;
}