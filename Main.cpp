// Basic inclusions.
#include <cstdlib>
#include <cstdio>
#include <iostream>

// MPI inclusions.
#include "mpi.h"

// Project inclusions.
#include "Exception.h"
#include "Lanczos.h"

// Main function :)
// Parameters are: N - matrix size. K - number of iterations.
int main(int argc, char* argv[]) {
	// MPI parameters.
	int p_rank, p_size;

	// Create exceptions.
	std::exception args;

	// Initialize MPI.
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p_size);

	// Show parameters.
	if (p_rank == 0) {
		std::cout << "Number of processes is: " << p_size << "\n";
		std::cout << "Matrix size is: " << argv[1] << "\n";
		std::cout << "Quantity of iterations is: " << argv[2] << "\n";
	}

	// Launch lanzcos iterations.
	try {
		// Set parameters first.
		int mat_size = atoi(argv[1]);
		int iter_num = atoi(argv[2]);

		// Check correctness.
		if ((iter_num > mat_size) || (p_rank > mat_size) || ((mat_size % p_size) != 0)) {
			throw args;
		}

		// Iterations themselves.
		double total_time = Lanczos(mat_size, mat_size / p_size, iter_num, p_rank, p_size);
		if (p_rank == 0) { std::cout << "Algorithm execution time is: " << total_time << "\n"; }
	}
	catch (invalid_args& args) {
		std::cout << args.get();
	}
	catch (...) {
		std::cout << "Critical error on process " << p_rank << "\n";
	}

	// Finalize MPI.
	MPI_Finalize();

	// Return :)
	return 0;
}