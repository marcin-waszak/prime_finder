#include "PrimeCuda.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

typedef unsigned int number_t;

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

__global__ void primes_in_range(number_t a, number_t b, bool *primes)
{
	const number_t number = a + (blockIdx.x * blockDim.x) + threadIdx.x;
	if (number > b)
	{
		return;
	}

  primes[number]=check_prime(number);

}


namespace CudaWrapper {
	std::list<number_t> cuda_wrapper(number_t a, number_t b)
	{

    thrust::host_vector<bool> primes_host(b-a+2);
    thrust::fill(primes_host.begin(), primes_host.end(), false);
    thrust::device_vector<bool> primes = primes_host;
    bool* d_primes =  thrust::raw_pointer_cast(&primes[0]);

    primes_in_range<<<(b-a)/1000+1, 1024>>>(a, b, d_primes);
  	cudaDeviceSynchronize();

    thrust::copy(primes.begin(), primes.end(), primes_host.begin());

    std::list<number_t> primes_list;

    for (number_t i = a; i <= b; ++i)
      if (primes_host[i]){
        primes_list.push_back(i);
      }
    return primes_list;

	}
}
