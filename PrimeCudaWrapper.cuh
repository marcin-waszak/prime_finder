#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Prime.h"

#include <stdio.h>

namespace CudaWrapper {
	number_t cuda_wrapper(number_t border_a, number_t border_b);
}