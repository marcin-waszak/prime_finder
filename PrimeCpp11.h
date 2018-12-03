#ifndef PRIME_FINDER_PRIMECPP11_H
#define PRIME_FINDER_PRIMECPP11_H

#include "Prime.h"

#include <thread>
#include <mutex>
#include <atomic>

class PrimeCpp11 : public Prime {
public:
	PrimeCpp11(number_t a, number_t b);
	int Find() override;

private:
	static void Worker(PrimeCpp11* instance);

	std::atomic_ullong current_;
	std::mutex mutex_;
};

#endif //PRIME_FINDER_PRIMECPP11_H
