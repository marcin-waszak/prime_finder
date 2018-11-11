#ifndef PRIME_FINDER_PRIME_H
#define PRIME_FINDER_PRIME_H

#include <list>

typedef unsigned long number_t;

class Prime {
public:
	Prime();
	static bool Check(number_t n);
	virtual void Find(number_t a, number_t b);
	void Print() const;

protected:
	std::list<number_t> primes_list;
};

#endif //PRIME_FINDER_PRIME_H
