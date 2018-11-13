#ifndef PRIME_FINDER_PRIMEOMP_H
#define PRIME_FINDER_PRIMEOMP_H

#include "Prime.h"

#include <omp.h>
#include <atomic>

class PrimeOmp : public Prime {
public:
	PrimeOmp(number_t a, number_t b);
	int Find() override;
};

#endif //PRIME_FINDER_PRIMEOMP_H
