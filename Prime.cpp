#include "Prime.h"

#include <cmath>
#include <cstdio>

Prime::Prime(number_t a, number_t b)
		:	border_a_(a),
			border_b_(b),
			found_(0) {
	primes_list.clear();
}

bool Prime::Check(number_t n) {
	for (number_t i = 2; i <= sqrt(n); ++i)
		if (n % i == 0)
			return false;

	return true;
}

void Prime::Print() const {
//	for (auto &number : primes_list)
//		printf("# %lu\n", number);
	printf("Found Prime: %lu\n", found_);
}

int Prime::Find() {
	for(number_t i = border_a_; i <= border_b_; ++i)
		if (Check(i))
			primes_list.push_back(i);

	found_ = primes_list.size();
	return 0;
}