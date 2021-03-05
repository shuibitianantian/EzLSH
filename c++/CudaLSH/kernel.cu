#pragma once
#include<iostream>
#include<math.h>


__global__ void matMulKernel(
    float *a, int a_rows, int a_cols, 
    float *b, int b_rows, int b_cols,
    float *c, int c_rows, int c_cols
)
{
	float Cvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	
    for (int i = 0; i < a_cols; ++i)
	{
		Cvalue += a[row * a_cols + i] * b[i * b_cols + col];
	}

    c[row * c_cols + col] = Cvalue;
}

// array A is the input matrix
// array B is the projection matrix
// array C is the results matrix

__global__ void hash_(
    float *A, int a_rows, int a_cols, 
    float *B, int b_rows, int b_cols,
    float *C, int c_rows, int c_cols, float* bits
)
{
	float Cvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
    
    int table_index = row / a_rows;
    int input_index = row % a_rows;

    for (int i = 0; i < a_cols; ++i)
	{
		Cvalue += A[input_index * a_cols + i] * B[table_index * (b_rows * b_cols) + i * b_cols + col];
	}

    C[row * c_cols + col] = std::signbit(-1 * Cvalue) * bits[col];
    // C[row * c_cols + col] = Cvalue;
}


__global__ void vec_sum(float* a, float* b, int cols) {

    int row = threadIdx.x + blockIdx.x * blockDim.x;

    for (int j = 0; j < cols; ++j) {
        b[row] += a[row * cols + j];
    }
    
}