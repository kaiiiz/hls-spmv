#include "spmv.h"

void spmv(int rowPtr[NUM_ROWS+1], int columnIndex[NNZ],
		DTYPE values[NNZ], DTYPE y[SIZE], DTYPE x[SIZE])
{
L1: for (int i = 0; i < NUM_ROWS; i++) {
		DTYPE y0 = 0;
	L2: for (int k = rowPtr[i]; k < rowPtr[i+1]; k++) {
#pragma HLS pipeline
			y0 += values[k] * x[columnIndex[k]];
		}
		y[i] = y0;
	}
}


void mv(DTYPE A[SIZE][SIZE], DTYPE y[SIZE], DTYPE x[SIZE])
{
	for (int i = 0; i < SIZE; i++) {
		DTYPE y0 = 0;
		for (int j = 0; j < SIZE; j++) {
#pragma HLS pipeline
			y0 += A[i][j] * x[j];
		}
		y[i] = y0;
	}
}

