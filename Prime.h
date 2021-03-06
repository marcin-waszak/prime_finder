#ifndef PRIME_FINDER_PRIME_H
#define PRIME_FINDER_PRIME_H

#include <list>

typedef unsigned int number_t;

class Prime {
public:
	Prime(number_t a, number_t b);
	static bool Check(number_t n);
	virtual int Find();
	void Print() const;

protected:
	number_t border_a_;
	number_t border_b_;
	std::list<number_t> primes_list;
	std::size_t found_;
};

#endif //PRIME_FINDER_PRIME_H
