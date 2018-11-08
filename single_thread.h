#ifndef PRIME_FINDER_SINGLE_THREAD_H
#define PRIME_FINDER_SINGLE_THREAD_H

#include <set>
#include <cmath>

#include "main.h"

bool primeCheck(number_t n);
void primeFind(std::set<number_t>* prime_set, number_t a, number_t b);

#endif //PRIME_FINDER_SINGLE_THREAD_H
