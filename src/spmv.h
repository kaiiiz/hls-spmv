#ifndef __SPMV_H__
#define __SPMV_H__

const static int SIZE = 256; // SIZE of square matrix
const static int NNZ = 3277; //Number of non-zero elements
const static int NUM_ROWS = 256;// SIZE;
typedef float DTYPE;
void spmv(int rowPtr[NUM_ROWS+1], int columnIndex[NNZ],
		  DTYPE values[NNZ], DTYPE y[SIZE], DTYPE x[SIZE]);
void mv(DTYPE A[SIZE][SIZE], DTYPE y[SIZE], DTYPE x[SIZE]);

#endif // __MATRIXMUL_H__ not defined
