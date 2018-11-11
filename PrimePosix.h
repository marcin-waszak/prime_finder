#ifndef PRIME_FINDER_PRIMEPOSIX_H
#define PRIME_FINDER_PRIMEPOSIX_H

#include "Prime.h"

#include <pthread.h>
#include <atomic>

#define NUM_THREADS 8

class PrimePosix : public Prime {
public:
	PrimePosix(number_t a, number_t b);
	int Find() override;

private:
	static void* Worker(void* instance);

	std::atomic_ullong current_;
	pthread_mutex_t mutex_;
};

#endif //PRIME_FINDER_PRIMEPOSIX_H
