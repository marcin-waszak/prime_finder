#include <cstdlib>
#include <cstdio>

#include "Prime.h"

int main(int argc, char** argv) {
	if (argc < 3)
		return 1;

	number_t a = strtoul(argv[1], nullptr, 0);
	number_t b = strtoul(argv[2], nullptr, 0);

	Prime finder;

	finder.Find(a, b);
	finder.Print();

	return 0;
}