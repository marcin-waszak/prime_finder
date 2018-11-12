#include "PrimeCpp11.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

PrimeCpp11::PrimeCpp11(number_t a, number_t b)
		:	Prime(a, b),
			current_(a) {
}

int PrimeCpp11::Find() {
	std::vector<std::thread> threads(NUM_THREADS);

	for (auto &thread : threads)
		thread = std::thread(Worker, this);

	for (auto &thread : threads)
		thread.join();

	found_ += primes_list.size();
	//printf("Found Prime: %lu\n", found_);
	return 0;
}

void PrimeCpp11::Worker(PrimeCpp11* instance) {
	PrimeCpp11* thiz = instance;
	std::list<number_t> numbers;

	while(1) {
		number_t n = thiz->current_.fetch_add(1);

		if (n > thiz->border_b_)
			break;

		if (Check(n))
			numbers.push_back(n);
	}

	thiz->mutex_.lock();
//	for(auto &n : numbers)
//		fprintf(stderr, "%lu\n", n);
//	fprintf(stderr, "L:%zu\n", numbers.size());
	thiz->primes_list.merge(numbers);
	thiz->mutex_.unlock();
}
