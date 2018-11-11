#include "PrimePosix.h"
#include "Prime.h"

#include <cstdio>
#include <cstdlib>

PrimePosix::PrimePosix(number_t a, number_t b)
		:	Prime(a, b),
			current_(a),
			mutex_(PTHREAD_MUTEX_INITIALIZER) {
}

int PrimePosix::Find() {
	pthread_t threads[NUM_THREADS];

	for (int rc, i = 0; i < NUM_THREADS; ++i) {
		if ((rc = pthread_create(&threads[i], NULL, PrimePosix::Worker, this))) {
			fprintf(stderr, "error: pthread_create, rc: %d\n", rc);
			return EXIT_FAILURE;
		}
	}

	for (int i = 0; i < NUM_THREADS; ++i)
		pthread_join(threads[i], nullptr);

	found_ += primes_list.size();
	//printf("Found Prime: %lu\n", found_);
	return 0;
}

void* PrimePosix::Worker(void* instance) {
	PrimePosix* thiz = (PrimePosix*)instance;
	std::list<number_t > numbers;

	while(1) {
		number_t n = thiz->current_.fetch_add(1);

		if (n > thiz->border_b_)
			break;

		if (Check(n))
			numbers.push_back(n);
	}

	pthread_mutex_lock(&thiz->mutex_);
//	for(auto &n : numbers)
//		fprintf(stderr, "%lu\n", n);
//	fprintf(stderr, "L:%zu\n", numbers.size());
	thiz->primes_list.merge(numbers);
	pthread_mutex_unlock(&thiz->mutex_);

	pthread_exit(nullptr);
}
