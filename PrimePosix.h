#ifndef PRIME_FINDER_PRIMEPOSIX_H
#define PRIME_FINDER_PRIMEPOSIX_H

#include "Prime.h"

#include <pthread.h>

#define NUM_THREADS 4

class PrimePosix : public Prime {
	PrimePosix();
	void Find(number_t a, number_t b) override;

private:
	void SafeInsert(number_t n);
	static void* Worker(void* context);

	pthread_mutex_t mutex_;
};


typedef struct {
	PrimePosix* instance;
	number_t a;
	number_t b;
} data_t;


#endif //PRIME_FINDER_PRIMEPOSIX_H
