#include "PrimeMpi.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mpi.h>

using namespace std;

PrimeMpi::PrimeMpi(number_t a, number_t b)
		:	Prime(a, b) {

}

int PrimeMpi::Find() {
    int world_rank; /* task identifier */
    int world_size; /* number of tasks */
    MPI_Init(NULL, NULL);

    // find out which process are we in
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // find out how many processes are there
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // every task will do its own portion of data
    // allocating memory for worst case where all numbers for task is prime 
    int* taskPrimes = new int[(border_b_ - border_a_) / world_size];
    int primesNum = 0;
    for (number_t n = border_a_ + world_rank; n <= border_b_; n = n + world_size) 
        if (Check(n)) 
            taskPrimes[primesNum++] = n;

    // if not master -> send found primes to master (master rank is 0)
    if (world_rank) {
        MPI_Send(&primesNum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	MPI_Send(taskPrimes, primesNum, MPI_INT, 0, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
    }
    else { // gather data in master (rank 0)
        MPI_Barrier(MPI_COMM_WORLD); // wait for other task to do their job       
	int taskPrimesNum;
        int* recTaskPrimes; //temp variable for primes recived from other task
        for (int i = 0; i < world_size; i++) {
            if (i) { // primes from other tasks
                MPI_Recv(&taskPrimesNum, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                recTaskPrimes = new int[taskPrimesNum];
                MPI_Recv(recTaskPrimes, taskPrimesNum, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < taskPrimesNum; j++)    	
				    primes_list.push_back(recTaskPrimes[j]);
			    delete recTaskPrimes;
            }
            else { // add primes from master
                for (int j = 0; j < primesNum; j++)
                    primes_list.push_back(taskPrimes[j]);
            }
        }
        found_ = primes_list.size();
        Print();
    }
    delete taskPrimes;

    MPI_Finalize();
	return 0;
}

int main(int argc, char** argv) {
	if (argc < 3)
		return 1;

	number_t a = strtoul(argv[1], nullptr, 0);
	number_t b = strtoul(argv[2], nullptr, 0);

	PrimeMpi finder_mpi(a, b);

	finder_mpi.Find();

	return 0;
}
