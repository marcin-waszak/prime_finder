#include "PrimePosix.h"
#include "Prime.h"

PrimePosix::PrimePosix(number_t a, number_t b)
		:	Prime(a, b),
			mutex_(PTHREAD_MUTEX_INITIALIZER),
			current_(a) {

}

void PrimePosix::Find() {
	for (short i = 0; i < NUM_THREADS; ++i) {

	}
}

//pthread_mutex_unlock(&mutex_);

void* PrimePosix::Worker(void* instance) {
	PrimePosix* thiz = (PrimePosix*)instance;

//	for(number_t i = thiz->border_a_; i <= thiz->border_b_; ++i)
//		if (Check(i))
//			thiz->SafeInsert(i);

	pthread_exit(nullptr);
}
