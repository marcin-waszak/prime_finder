#include "PrimePosix.h"
#include "Prime.h"

PrimePosix::PrimePosix()
		: mutex_(PTHREAD_MUTEX_INITIALIZER) {

}

void PrimePosix::Find(number_t a, number_t b) {
	data_t threads_data[NUM_THREADS];

	for (short i = 0; i < NUM_THREADS; ++i) {
		threads_data[i].instance = this;
	}
}

void PrimePosix::SafeInsert(number_t n) {
	pthread_mutex_lock(&mutex_);
	primes_list.push_back(n);
	pthread_mutex_unlock(&mutex_);
}

void* PrimePosix::Worker(void* context) {
	data_t* data = (data_t*)context;
	PrimePosix* thiz = data->instance;

	for(number_t i = data->a; i <= data->b; ++i)
		if (Check(i))
			thiz->SafeInsert(i);

	pthread_exit(nullptr);
}
