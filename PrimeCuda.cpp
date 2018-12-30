#include "PrimeCuda.h"
#include "PrimeCudaWrapper.cuh"

PrimeCuda::PrimeCuda(number_t a, number_t b)
		:	Prime(a, b) {

}

int PrimeCuda::Find() {

  found_ = CudaWrapper::cuda_wrapper(border_a_, border_b_);

	return 0;
}
