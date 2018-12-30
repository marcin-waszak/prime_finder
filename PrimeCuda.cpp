#include "PrimeCuda.h"
#include "PrimeCuda.cuh"

PrimeCuda::PrimeCuda(number_t a, number_t b)
		:	Prime(a, b) {

}

int PrimeCuda::Find() {

  primes_list = CudaWrapper::cuda_wrapper(border_a_, border_b_);
	found_ = primes_list.size();

	return 0;
}
