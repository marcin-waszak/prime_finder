#ifndef PRIME_FINDER_PRIMEPOSIX_H
#define PRIME_FINDER_PRIMEPOSIX_H

#include "Prime.h"

#include <pthread.h>
#include <atomic>

#define NUM_THREADS 4

class PrimePosix : public Prime {
	PrimePosix(number_t a, number_t b);
	void Find() override;

private:
	static void* Worker(void* instance);

	pthread_mutex_t mutex_;
	std::atomic_ullong current_;
};

#endif //PRIME_FINDER_PRIMEPOSIX_H
