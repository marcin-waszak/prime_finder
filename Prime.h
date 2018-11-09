#ifndef PRIME_FINDER_PRIME_H
#define PRIME_FINDER_PRIME_H

#include <set>

typedef unsigned long number_t;

class Prime {
public:
	Prime();
	bool Check(number_t n);
	virtual void Find(number_t a, number_t b);
	void Print();

private:
	std::set<number_t> prime_set_;
};

#endif //PRIME_FINDER_PRIME_H
