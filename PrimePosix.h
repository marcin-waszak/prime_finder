#ifndef PRIME_FINDER_PRIMEPOSIX_H
#define PRIME_FINDER_PRIMEPOSIX_H

#include "Prime.h"

#include <pthread.h>

class PrimePosix : public Prime {
public:
	PrimePosix(number_t a, number_t b);
	int Find() override;

private:
	static void* Worker(void* instance);

	number_t current_;
	pthread_mutex_t mutex_;
};

#endif //PRIME_FINDER_PRIMEPOSIX_H
