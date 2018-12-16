#ifndef PRIME_FINDER_PRIMEMPI_H
#define PRIME_FINDER_PRIMEMPI_H

#include "Prime.h"

#include <mpi.h>
#include <atomic>

class PrimeMpi : public Prime {
public:
	PrimeMpi(number_t a, number_t b, int* argc, char*** argv);
	int Find() override;

private:
	int* argc_;
	char*** argv_;
};

#endif //PRIME_FINDER_PRIMEMPI_H