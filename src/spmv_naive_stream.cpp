#include "spmv.h"
#include <hls_stream.h>

void spmv_kernel(int rows_length[NUM_ROWS], int cols[NNZ], DTYPE values[NNZ], DTYPE y[SIZE], DTYPE x[SIZE])
{
#pragma HLS DATAFLOW

	hls::stream<int>   rows_fifo;
	hls::stream<DTYPE> values_fifo;
	hls::stream<int>   cols_fifo;
	hls::stream<DTYPE> results_fifo;


	for (int i = 0; i < NUM_ROWS; i++) {
#pragma HLS PIPELINE
		rows_fifo << rows_length[i];
	}

	for (int i = 0; i < NNZ; i++) {
#pragma HLS PIPELINE
		values_fifo << values[i];
		cols_fifo   << cols[i];
	}

	int col_left = 0;
	DTYPE sum = 0;
	DTYPE value;
	int col;

	for (int i = 0; i < NNZ; i++) {
#pragma HLS PIPELINE
		if (col_left == 0) {
			col_left = rows_fifo.read();
			sum = 0;
		}
		value = values_fifo.read();
		col   = cols_fifo.read();
		sum  += value * x[col];
		col_left--;
		if (col_left == 0) {
			results_fifo << sum;
		}
	}

	for (int i = 0; i < NUM_ROWS; i++) {
#pragma HLS PIPELINE
		y[i] = results_fifo.read();
	}
}


void spmv(int rowPtr[NUM_ROWS + 1], int cols[NNZ], DTYPE values[NNZ], DTYPE y[SIZE], DTYPE x[SIZE])
{
	// rowPtr to rows_length
	int rows_length[NUM_ROWS] = {0};
	for (int i = 1; i < NUM_ROWS + 1; i++) {
#pragma HLS PIPELINE
		rows_length[i - 1] = rowPtr[i] - rowPtr[i - 1];
	}

	spmv_kernel(rows_length, cols, values, y, x);
}
