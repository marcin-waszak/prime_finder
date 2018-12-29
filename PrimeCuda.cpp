#include "PrimeCuda.h"
#include "PrimeCuda.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

PrimeCuda::PrimeCuda(number_t a, number_t b)
		:	Prime(a, b) {

}

int PrimeCuda::Find() {

  found_ = Wrapper::wrapper(border_a_, border_b_);

	return 0;
}
