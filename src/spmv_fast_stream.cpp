#include "spmv.h"
#include <hls_stream.h>

#define II 9

void spmv_kernel(
		int rows_length[NUM_ROWS],
		int rows_length_pad[NUM_ROWS],
		int cols[NNZ],
		DTYPE values[NNZ],
		DTYPE y[SIZE],
		DTYPE x[SIZE],
		int new_nnz)
{
#pragma HLS DATAFLOW

	int row_length_pad = 0, row_length = 0, k = 0, row_counter = 0;
	hls::stream<DTYPE> values_fifo;
	hls::stream<int>   cols_fifo;
	hls::stream<DTYPE> results_fifo;

	DTYPE sum = 0;
	DTYPE value;
	int col;
	DTYPE term[II];

	for (int i = 0; i < NNZ; i++) {
#pragma HLS PIPELINE
		values_fifo << values[i];
		cols_fifo   << cols[i];
	}

	for (int i = 0; i < new_nnz; i+=II) {
#pragma HLS PIPELINE
		if (row_length_pad == 0) {
			row_length_pad = rows_length_pad[k];
			row_length = rows_length[k++];
			row_counter = 0;
			sum = 0;
		}

		for (int j = 0; j < II; j++) {
			row_counter++;
			if (row_counter > row_length) {
				term[j] = 0;
			} else {
				value = values_fifo.read();
				col   = cols_fifo.read();
				term[j] = value * x[col];
			}
		}

		DTYPE sum_tmp = 0;
		for (int j = 0; j < II; j++) {
			sum_tmp += term[j];
		}
		sum += sum_tmp;

		row_length_pad -= II;
		if (row_length_pad == 0) {
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

	int rows_length_pad[NUM_ROWS];
	int new_nnz = 0;
	for (int i = 0; i < NUM_ROWS; i++) {
#pragma HLS PIPELINE
		int r = rows_length[i];
		int r_diff = r % II;
		if (r == 0) {
			rows_length_pad[i] = II;
			new_nnz += II;
		} else if (r_diff != 0) {
			rows_length_pad[i] = r + (II - r_diff);
			new_nnz += r + (II - r_diff);
		} else {
			rows_length_pad[i] = r;
			new_nnz += r;
		}
	}

	spmv_kernel(rows_length, rows_length_pad, cols, values, y, x, new_nnz);
}
