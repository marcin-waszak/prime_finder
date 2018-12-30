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
	// Corner cases
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

void Prime::Print() const {
// Uncomment following in order to print prime numbers

//	for (auto &number : primes_list)
//		printf("# %lu\n", number);
	printf("Primes found: %u\n", found_);
}

int Prime::Find() {
	for(number_t i = border_a_; i <= border_b_; ++i)
		if (Check(i))
			primes_list.push_back(i);

	found_ = primes_list.size();
	return 0;
}
