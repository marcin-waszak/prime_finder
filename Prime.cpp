#include "Prime.h"

#include <cmath>
#include <cstdio>

Prime::Prime() {
	prime_set_.clear();
}

bool Prime::Check(number_t n) {
	for (number_t i = 2; i <= sqrt(n); ++i)
		if (n % i == 0)
			return false;

	return true;
}

void Prime::Print() {
	for (auto &number : prime_set_)
		printf("%lu\n", number);
}

void Prime::Find(number_t a, number_t b) {
	for(number_t i = a; i <= b; ++i)
		if (Check(i))
			prime_set_.insert(i);
}
