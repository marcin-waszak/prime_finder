#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <iostream>

#include "PrimePosix.h"

int main(int argc, char** argv) {
	if (argc < 3)
		return 1;

	number_t a = strtoul(argv[1], nullptr, 0);
	number_t b = strtoul(argv[2], nullptr, 0);

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	Prime finder(a, b);

	finder.Find();
	finder.Print();

	fprintf(stderr, "Posix:\n");

	std::chrono::steady_clock::time_point inter = std::chrono::steady_clock::now();
	PrimePosix finder_posix(a, b);
	
	finder_posix.Find();
	finder_posix.Print();

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << "SINGLE: " << std::chrono::duration_cast<std::chrono::milliseconds>(inter - begin).count() <<std::endl;
	std::cout << "POSIX:  " << std::chrono::duration_cast<std::chrono::milliseconds>(end - inter).count() <<std::endl;


	return 0;
}