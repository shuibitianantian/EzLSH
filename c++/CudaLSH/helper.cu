#pragma once

#include<iostream>
#include<random>

void initialize_projection_plane(float* m, int numbers, float sigma=1, float mu=0){
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1, 1.0);
    
    for (int i = 0; i < numbers; ++i) {
        m[i] = sigma * distribution(generator) + mu;
    }
}

void initialize_bits(float* bits, int hashSize){
    bits[0] = 1.0;
    for (int i = 1; i < hashSize; ++i) {
        bits[i] = 2 * bits[i-1];
    }
}

void show_array(float* a, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << a[i*cols + j] << ' ';
        }
        std::cout << std::endl;
    }
}

// int main(){
    
//     int tableNums = 4;
//     int inputDim = 100;
//     int hashSize = 12;
//     int inputSize = 5000000;
    
//     // Initialize the inputs on host
//     int input_bytes = inputSize * inputDim * sizeof(float);
//     float* input;
//     input = (float*) malloc(input_bytes);
//     initialize_projection_plane(input, input_bytes / sizeof(float));

//     // Initialize projection planes
//     float* projections;
//     int projection_plane_bytes = tableNums * inputDim * hashSize * sizeof(float);

//     projections = (float*) malloc(projection_plane_bytes);
//     initialize_projection_plane(projections, projection_plane_bytes / sizeof(float));

//     // Initialize bits
//     float* bits = (float*) malloc(hashSize * sizeof(float));    
//     initialize_bits(bits, hashSize);

//     // initialize result
//     int results_bytes = tableNums * inputSize * hashSize * sizeof(float);
//     float* results = (float*) malloc(results_bytes);

//     // initialize hash values
//     int hash_values_bytes = tableNums * inputSize * sizeof(float);
//     float* hash_values = (float*) malloc(hash_values_bytes);
//     std::fill(hash_values, hash_values + hash_values_bytes / sizeof(float), (float) 0.0);

//     auto start = high_resolution_clock::now(); 
    
//     index(input, inputSize, inputDim, projections, tableNums, hashSize, bits, hash_values);
    
//     auto stop = high_resolution_clock::now(); 
//     cout <<  duration_cast<microseconds>(stop - start).count() << endl;

//     // show_array(projections, tableNums * inputDim, hashSize);
//     // show_array(input, inputSize, inputDim);
//     // show_array(results, inputSize, tableNums * hashSize);
//     // show_array(hash_values, inputSize * tableNums, 1);

//     free(input);
//     free(projections);
//     free(bits);
//     free(hash_values);

//     return 0;
// }