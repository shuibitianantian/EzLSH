#include<cuda_runtime.h>
#include "kernel.cu"
#include<iostream>

#include<chrono> 
#include<vector>
#include<string.h>
#include "helper.cu"

using namespace std;
using namespace std::chrono; 


void index(float* input, int inputSize, int inputDim, // scale of input
    float* projections, int tableNums, int hashSize, // scale of projection planes
    float* bits, // bits for hash
    float* hash_values){

    /*
    initialize gpu memory
    */

    // Initialize the inputs on device
    int input_bytes = inputSize * inputDim * sizeof(float);
    float* gpu_input;
    cudaMalloc(&gpu_input, input_bytes);
    cudaMemcpy(gpu_input, input, input_bytes, cudaMemcpyHostToDevice);

    // Initialize projection planes on device
    int projection_plane_bytes = tableNums * inputDim * hashSize * sizeof(float);
    float* gpu_projections;
    cudaMalloc(&gpu_projections, projection_plane_bytes);
    cudaMemcpy(gpu_projections, projections, projection_plane_bytes, cudaMemcpyHostToDevice);

    // Initialize bits on device
    float* gpu_bits;
    int bits_bytes = hashSize * sizeof(float);
    cudaMalloc(&gpu_bits, bits_bytes);
    cudaMemcpy(gpu_bits, bits, bits_bytes, cudaMemcpyHostToDevice);

    // initialize result
    int results_bytes = tableNums * inputSize * hashSize * sizeof(float);
    float* results = (float*) malloc(results_bytes);
    float* gpu_results;
    cudaMalloc(&gpu_results, results_bytes);

    // initialize hash values
    int hash_values_bytes = tableNums * inputSize * sizeof(float);
    float* gpu_hash_values;
    cudaMalloc(&gpu_hash_values, hash_values_bytes);
    cudaMemcpy(gpu_hash_values, hash_values, hash_values_bytes, cudaMemcpyHostToDevice);

    /*
    Start indexing
    */

    dim3 blockSize(2, 512);
    dim3 gridSize((hashSize * tableNums + blockSize.x - 1) / blockSize.x, (inputSize + blockSize.y - 1) / blockSize.y);
    hash_<<<gridSize, blockSize>>>(gpu_input, inputSize, inputDim, gpu_projections, inputDim, hashSize, gpu_results, inputSize, hashSize * tableNums, gpu_bits);

    cudaDeviceSynchronize();
    cudaError_t cudaStatus = cudaGetLastError();
    cout << "Hashing : " << cudaGetErrorString(cudaStatus) << endl;

    dim3 blockSizeSum(1, 1024);
    dim3 gridSizeSum(tableNums, (inputSize + blockSizeSum.y - 1) / blockSizeSum.y);
    vec_sum<<<gridSizeSum, blockSizeSum>>>(gpu_results, gpu_hash_values, inputSize, tableNums, hashSize);
    
    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    cout << "Sum bits: " << cudaGetErrorString(cudaStatus) << endl;

    cudaMemcpy(hash_values, gpu_hash_values, hash_values_bytes, cudaMemcpyDeviceToHost);

    free(results);

    cudaFree(gpu_input);
    cudaFree(gpu_projections);
    cudaFree(gpu_bits);
    cudaFree(gpu_results);
    cudaFree(gpu_hash_values);
}

// void indexing(float* input, int inputSize, int inputDim, // scale of input
//     float* projections, int tableNums, int hashSize, // scale of projection planes
//     float* bits, // bits for hash
//     float* hash_values) {
//     index(input, inputSize, inputDim, projections, tableNums, hashSize, bits, hash_values);
// }



