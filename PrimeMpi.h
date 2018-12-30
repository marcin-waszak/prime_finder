#ifndef PRIME_FINDER_PRIMEMPI_H
#define PRIME_FINDER_PRIMEMPI_H

#include "Prime.h"

#include <mpi.h>

class PrimeMpi : public Prime {
public:
	PrimeMpi(number_t a, number_t b);
	int Find() override;
};

#endif //PRIME_FINDER_PRIMEMPI_H