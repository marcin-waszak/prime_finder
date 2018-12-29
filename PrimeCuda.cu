#include "PrimeCuda.cuh"

typedef unsigned long number_t;

__device__ bool check_prime(number_t n)
{
  if (n <= 1)
		return false;
	if (n <= 3)
		return true;

	// This is checked so that we can skip
	// middle five numbers in below loop
	if (n % 2 == 0 || n % 3 == 0)
		return false;

	for (number_t i = 5; i*i <= n; i += 6)
		if(n % i == 0 || n % (i + 2) == 0)
			return false;

	return true;
}

__global__ void primes_in_range(number_t a, number_t b, int *result)
{
	const number_t number = a + (blockIdx.x * blockDim.x) + threadIdx.x;
	if (number > b)
	{
		return;
	}

  if (check_prime(number))
    atomicAdd(result, 1);
}


namespace CudaWrapper {
	int cuda_wrapper(number_t a, number_t b)
	{

    int *result;
  	cudaMallocManaged(&result, 4);
  	*result = 0;

    primes_in_range<<<(b-a)/1000+1, 1024>>>(a, b, result);
  	cudaDeviceSynchronize();

    return *result;

	}
}
