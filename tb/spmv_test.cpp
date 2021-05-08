#include "../src/spmv.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

DTYPE M[SIZE][SIZE] = {0};
DTYPE x[SIZE] = {0};
DTYPE values[NNZ] = {0};
int columnIndex[NNZ] = {0};
int rowPtr[NUM_ROWS+1] = {0};
DTYPE y_sw[SIZE] = {0};
DTYPE y[SIZE] = {0};

void matrixvector(DTYPE A[SIZE][SIZE], DTYPE *y, DTYPE *x)
{
	for (int i = 0; i < SIZE; i++) {
		DTYPE y0 = 0;
		for (int j = 0; j < SIZE; j++)
			y0 += A[i][j] * x[j];
		y[i] = y0;
	}
}

void load_matrix()
{
	ifstream matrix("./matrix.dat");
	string line;
	int tmp, i = 0, j = 0;
	while (getline(matrix, line)) {
		istringstream ss(line);
		while (ss >> tmp) {
			if (tmp > 0) {
				M[i][j] = tmp;
			}
			j++;
		}
		i++;
		j = 0;
	}
	matrix.close();
}

void load_data()
{
	ifstream data("./data.dat");
	string line;
	int tmp, i = 0;

	getline(data, line);
	istringstream ss(line);
	while (ss >> tmp) {
		values[i++] = tmp;
	}

	data.close();
}

void load_rows()
{
	ifstream rows("./rows.dat");
	string line;
	int tmp, i = 0;

	getline(rows, line);
	istringstream ss(line);
	while (ss >> tmp) {
		rowPtr[i++] = tmp;
	}

	rows.close();
}

void load_cols()
{
	ifstream cols("./cols.dat");
	string line;
	int tmp, i = 0;

	getline(cols, line);
	istringstream ss(line);
	while (ss >> tmp) {
		columnIndex[i++] = tmp;
	}

	cols.close();
}

void gen_input()
{
	for (int i = 0; i < SIZE; i++) {
		x[i] = rand() % 100;
	}
}


int main(){
	int fail = 0;
	load_matrix();
	load_data();
	load_rows();
	load_cols();
	gen_input();

	spmv(rowPtr, columnIndex, values, y, x);
	matrixvector(M, y_sw, x);

	for(int i = 0; i < SIZE; i++) {
		if(y_sw[i] != y[i])
			fail = 1;
	}

	if(fail == 1)
		cout << "FAILED\n";
	else
		cout << "PASS\n";

	return fail;
}
