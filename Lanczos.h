#pragma once
// Matrix numbers are set as same numbers, since we only want to
// check performance, while validation of results isn't our value.
// We don't even calculate actual eigenvalues.

// Basic inclusions.

// MPI and OpenMP inclusions.
#include "mpi.h"
#include <omp.h>

// Lanczos iterator.
double Lanczos(int mat_size, int mat_lines, int iter_num, int p_rank, int p_size) {
	// Preset common parameters.
	double* A_partition = new double[mat_lines * mat_size]; // Process-dependant.
	double* prev_lanz_vector = new double[mat_lines]; // Process-dependant.
	double* lanz_vector = new double[mat_lines]; // Process-dependant.
	double* next_lanz_vector = new double[mat_lines]; // Process-dependant.
	double* z_vector = new double[mat_lines]; // Process-dependant.

	double prev_beta; // Process-dependant, synchronizes.
	double cur_alpha; // Process-dependant, synchronizes.
	double cur_beta;  // Process-dependant, synchronizes.

	double* common_vector = new double[mat_size]; // Qi vector storage. Same on all procs.
	double* alpha_values = new double[iter_num]; // Same on all procs.
	double* beta_values = new double[iter_num]; // Same on all procs.

	// Time parameters for performance measurement.
	double start_time, end_time, total_time;

	// Start timer.
	start_time = MPI_Wtime();

	// Preset iteration parameters.
	prev_beta = 0; // beta[0] = 0;
	for (int i = 0; i < mat_lines; ++i) {
		prev_lanz_vector[i] = 0; // q0 = [0];
		lanz_vector[i] = 1.0 / (double)(mat_size); // q1 generated from a default b vector.
	}

	// Primary cycle.
	for (int i = 1; i <= iter_num; ++i) {
		// Calculate z vector in 3 steps - common vector assignment, common vector bcast and multiplication.
		// Vector assignment.
		#pragma omp parallel for
		for (int j = 0; j < mat_lines; ++j) {
			common_vector[mat_lines * p_rank + j] = lanz_vector[j];
		}
		// Vector broadcast.
		for (int j = 0; j < p_size; ++j) {
			MPI_Bcast(&common_vector[mat_lines * p_rank], mat_lines, MPI_DOUBLE, j, MPI_COMM_WORLD);
		}
		// Multiplication.
		#pragma omp parallel for
		for (int j = 0; j < mat_lines; ++j) { // Outer cycle.
			z_vector[j] = 0;
			for (int k = 0; k < mat_size; ++k) { // Inner cycle.
				z_vector[j] += A_partition[j * mat_size + k] * common_vector[k];
			}
		}

		// Calculate cur_alpha.
		cur_alpha = 0;
		for (int j = 0; j < mat_lines; ++j) {
			cur_alpha += lanz_vector[j] * z_vector[j];
		}
		MPI_Allreduce(MPI_IN_PLACE, &cur_alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		alpha_values[i - 1] = cur_alpha;

		// Recalculate z vector.
		#pragma omp parallel for
		for (int j = 0; j < mat_lines; ++j) {
			z_vector[j] = z_vector[j] - cur_alpha * lanz_vector[j] - prev_beta * prev_lanz_vector[j];
		}

		// Calculate cur_beta.
		cur_beta = 0;
		for (int j = 0; j < mat_lines; ++j) {
			cur_beta += z_vector[j] * z_vector[j];
		}
		MPI_Allreduce(MPI_IN_PLACE, &cur_beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		prev_beta = cur_beta; // Use on next iteration.
		beta_values[i - 1] = cur_beta;
		
		// Check cur_beta for not being zero.
		if (cur_beta == 0) {
			//break;
		}

		// Calculate next_lanz_vector.
		#pragma omp parallel for
		for (int j = 0; j < mat_lines; ++j) {
			if (cur_beta == 0) {
				next_lanz_vector[j] = z_vector[j];
			}
			else {
				next_lanz_vector[j] = z_vector[j] / cur_beta;
			}
		}

		// Set values for next iteration.
		prev_beta = cur_beta;
		#pragma omp parallel for
		for (int j = 0; j < mat_lines; ++j) {
			prev_lanz_vector[j] = lanz_vector[j];
			lanz_vector[j] = next_lanz_vector[j];
		}

		// Synchronize before further movement.*/
		MPI_Barrier(MPI_COMM_WORLD);
	}

	// Calculate total time.
	end_time = MPI_Wtime();
	total_time = end_time - start_time;
	MPI_Allreduce(MPI_IN_PLACE, &total_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	// Free memory and return.
	delete A_partition;
	delete prev_lanz_vector;
	delete lanz_vector;
	delete next_lanz_vector;
	delete z_vector;

	delete common_vector;
	delete alpha_values;
	delete beta_values;

	return total_time; 
}
