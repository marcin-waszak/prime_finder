#ifndef PRIME_FINDER_PRIMECUDA_H
#define PRIME_FINDER_PRIMECUDA_H

#include "Prime.h"

// #include <omp.h>
#include <atomic>

class PrimeCuda : public Prime {
public:
	PrimeCuda(number_t a, number_t b);
	int Find() override;
};

#endif //PRIME_FINDER_PRIMEOMP_H
