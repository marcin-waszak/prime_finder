#include "single_thread.h"

bool primeCheck(number_t n) {
	for (number_t i = 2; i <= sqrt(n); ++i)
		if (n % i == 0)
			return false;

	return true;
}typedef unsigned long number_t;

void primeFind(std::set<number_t>* prime_set, number_t a, number_t b) {
	for(number_t i = a; i <= b; ++i)
		if (primeCheck(i))
			prime_set->insert(i);
}
