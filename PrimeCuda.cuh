#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Prime.h"

#include <stdio.h>

namespace CudaWrapper {
	std::list<number_t> cuda_wrapper(number_t llimit, number_t ulimit);
}
