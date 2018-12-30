#include "PrimeMpi.h"

#include <cstdio>
#include <cstdlib>

PrimeMpi::PrimeMpi(number_t a, number_t b)
		:	Prime(a, b) {

}

int PrimeMpi::Find() {
	int world_rank; /* task identifier */
	int world_size; /* number of tasks */

	// find out which processor are we in
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// find out how many processes are there
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// every processor will do its own portion of data
	// allocating memory for worst case where all numbers for task is prime
	int proc_portions = world_size * world_size * world_size;
	number_t chunk_size = border_b_ / (world_size * proc_portions) + 1;
	number_t* task_primes = new number_t[chunk_size * proc_portions];
	int primes_num = 0;

	// checking if prime
	for (int portion_num = 0; portion_num < proc_portions; portion_num++) {
		number_t task_a = border_a_ + (world_rank + world_size * portion_num) * chunk_size; // start number of portion of data to check
		number_t task_b = std::min(border_b_, border_a_ + (world_rank + world_size * portion_num + 1) * chunk_size - 1); // end number of portion of data to check

		for (number_t n = task_a; n <= task_b; n++)
			if (Check(n))
				task_primes[primes_num++] = n;
	}

	// if not master -> send found primes to master (master rank is 0)
	if (world_rank) {
		MPI_Send(&primes_num, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
		MPI_Send(task_primes, primes_num, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
	} else { // gather data in master (rank 0)
		number_t task_primes_num;
		number_t* rec_task_primes; //temp variable for primes recived from other task
		for (int i = 0; i < world_size; i++) {
			if (i) { // primes from other tasks
				MPI_Recv(&task_primes_num, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				rec_task_primes = new number_t[task_primes_num];
				MPI_Recv(rec_task_primes, (int)task_primes_num, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				for (number_t j = 0; j < task_primes_num; j++)
					primes_list.push_back(rec_task_primes[j]);
				delete[] rec_task_primes;
			} else { // add primes from master
				for (int j = 0; j < primes_num; j++)
					primes_list.push_back(task_primes[j]);
			}
		}

		found_ = primes_list.size();
		Print();
	}

	delete[] task_primes;
	return 0;
}

int main(int argc, char** argv) {
	if (argc < 3)
		return 1;

	MPI_Init(NULL, NULL);

	number_t a = strtoul(argv[1], nullptr, 0);
	number_t b = strtoul(argv[2], nullptr, 0);

	PrimeMpi finder_mpi(a, b);

	finder_mpi.Find();
	MPI_Finalize();

	return 0;
}
