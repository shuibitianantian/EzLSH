#include<cuda_runtime.h>
#include "kernel.cu"
#include<iostream>
#include<random>
#include<chrono> 
#include<vector>
#include<string.h>


using namespace std;
using namespace std::chrono; 

void show_array(float* a, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << a[i*cols + j] << ' ';
        }
        cout << endl;
    }
}

void initialize_bits(float* bits, int hashSize){
    bits[0] = 1.0;
    for (int i = 1; i < hashSize; ++i) {
        bits[i] = 2 * bits[i-1];
    }
}

void initialize_projection_plane(float* m, int numbers){
    // default_random_engine generator;
    // uniform_real_distribution<double> distribution(0.0,1.0);
    
    for (int i = 0; i < numbers; ++i) {
        m[i] = i;
    }
}


int main(){

    
    int tableNums = 4;
    int inputDim = 500;
    int hashSize = 4;
    int inputSize = 600000;
    
    // Initialize the inputs on host
    int input_bytes = inputSize * inputDim * sizeof(float);
    float* input;
    input = (float*) malloc(input_bytes);
    initialize_projection_plane(input, input_bytes / sizeof(float));

    float* gpu_input;
    cudaMalloc(&gpu_input, input_bytes);

    // Initialize projection planes
    float* projections;
    float* gpu_projections;

    int projection_plane_bytes = tableNums * inputDim * hashSize * sizeof(float);

    projections = (float*) malloc(projection_plane_bytes);
    initialize_projection_plane(projections, projection_plane_bytes / sizeof(float));
    cudaMalloc(&gpu_projections, projection_plane_bytes);

    
    // Initialize bits
    float* bits = (float*) malloc(hashSize * sizeof(float));    
    initialize_bits(bits, hashSize);

    float* gpu_bits;
    int bits_bytes = hashSize * sizeof(float);
    cudaMalloc(&gpu_bits, bits_bytes);

    // initialize result
    int results_bytes = tableNums * inputSize * hashSize * sizeof(float);
    float* results = (float*) malloc(results_bytes);

    float* gpu_results;
    cudaMalloc(&gpu_results, results_bytes);
    

    // initialize hash values
    int hash_values_bytes = tableNums * inputSize * sizeof(float);
    float* hash_values = (float*) malloc(hash_values_bytes);
    std::fill(hash_values, hash_values + hash_values_bytes / sizeof(float), (float) 0.0);

    float* gpu_hash_values;
    cudaMalloc(&gpu_hash_values, hash_values_bytes);

    auto start = high_resolution_clock::now(); 

    // move input from host to device
    cudaMemcpy(gpu_input, input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_projections, projections, projection_plane_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_bits, bits, bits_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_hash_values, hash_values, hash_values_bytes, cudaMemcpyHostToDevice);

    dim3 blockSize(2, 512);
    dim3 gridSize((hashSize + blockSize.x - 1) / blockSize.x, (tableNums * inputSize + blockSize.y - 1) / blockSize.y);
    
    cout << "Block Size: " << blockSize.x << '*' << blockSize.y << endl;
    cout << "Grid Size: " << gridSize.x << '*' << gridSize.y << endl;
    
    cudaError_t cudaStatus = cudaGetLastError();
    cout << "Create architecture: " << cudaGetErrorString(cudaStatus) << endl;

    hash_<<<gridSize, blockSize>>>(gpu_input, inputSize, inputDim, gpu_projections, inputDim, hashSize, gpu_results, tableNums * inputSize, hashSize, gpu_bits);
    cudaDeviceSynchronize();

    cudaStatus = cudaGetLastError();
    cout << "Hashing : " << cudaGetErrorString(cudaStatus) << endl;
    
    vec_sum<<<(inputSize * tableNums + 1024 - 1) / 1024, 1024>>>(gpu_results, gpu_hash_values, hashSize);
    cudaDeviceSynchronize();

    cudaStatus = cudaGetLastError();
    cout << "Sum bits: " << cudaGetErrorString(cudaStatus) << endl;

    cudaMemcpy(hash_values, gpu_hash_values, hash_values_bytes, cudaMemcpyDeviceToHost);
    auto stop = high_resolution_clock::now(); 
    cout <<  duration_cast<microseconds>(stop - start).count() << endl;

    // show_array(projections, tableNums * inputDim, hashSize);
    // show_array(hash_values, tableNums * inputSize, 1);
    for (int i = 0; i < tableNums * inputSize; ++i) {
        if (hash_values[i] != 15) {
            cout << "ds" << endl;
        }
    }

    free(input);
    free(projections);
    free(bits);
    free(results);
    free(hash_values);

    cudaFree(gpu_input);
    cudaFree(gpu_projections);
    cudaFree(gpu_bits);
    cudaFree(gpu_results);
    cudaFree(gpu_hash_values);

    return 0;
}