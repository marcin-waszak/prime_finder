#include <cstdlib>
#include <cstdio>

#include "single_thread.h"

void primePrint(const std::set<number_t>& prime_set) {
	for (auto &number : prime_set)
		printf("%lu\n", number);
}

int main(int argc, char** argv) {
	if (argc < 3)
		return 1;

	number_t a = strtoul(argv[1], nullptr, 0);
	number_t b = strtoul(argv[2], nullptr, 0);

	std::set<number_t> prime_set;

	primeFind(&prime_set, a, b);
	primePrint(prime_set);

	return 0;
}