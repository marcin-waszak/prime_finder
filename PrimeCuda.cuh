#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Prime.h"

#include <stdio.h>

namespace Wrapper {
	void wrapper(number_t llimit, number_t ulimit);
}