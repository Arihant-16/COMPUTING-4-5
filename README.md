# COMPUTING-4-5
COMPUTING ASSIGNMENT 4&amp;5 



ASSIGNMENT 4 
Q1.
#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

// CUDA kernel where each thread performs a different task
__global__ void compute_sums(int *iterative_sum, int *formula_sum) {
    int tid = threadIdx.x;

    if (tid == 0) {
        // Iterative approach
        int sum = 0;
        for (int i = 1; i <= N; ++i) {
            sum += i;
        }
        *iterative_sum = sum;
    } else if (tid == 1) {
        // Direct formula approach
        *formula_sum = N * (N + 1) / 2;
    }
}

int main() {
    int h_iterative_sum = 0;
    int h_formula_sum = 0;
    int *d_iterative_sum, *d_formula_sum;  




Q2. 
#include <iostream>
#include <omp.h>
#include <vector>

void merge(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> leftSub(arr.begin() + left, arr.begin() + mid + 1);
    std::vector<int> rightSub(arr.begin() + mid + 1, arr.begin() + right + 1);

    int i = 0, j = 0, k = left;
    while (i < leftSub.size() && j < rightSub.size()) {
        arr[k++] = (leftSub[i] <= rightSub[j]) ? leftSub[i++] : rightSub[j++];
    }
    while (i < leftSub.size()) arr[k++] = leftSub[i++];
    while (j < rightSub.size()) arr[k++] = rightSub[j++];
}

void parallelMergeSort(std::vector<int>& arr, int left, int right, int depth = 0) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        if (depth < 4) { // Limit depth to prevent oversubscription
            #pragma omp parallel sections
            {
                #pragma omp section
                parallelMergeSort(arr, left, mid, depth + 1);
                #pragma omp section
                parallelMergeSort(arr, mid + 1, right, depth + 1);
            }
        } else {
            parallelMergeSort(arr, left, mid, depth + 1);
            parallelMergeSort(arr, mid + 1, right, depth + 1);
        }

        merge(arr, left, mid, right);
    }
}

int main() {
    std::vector<int> data(1000);
    // Initialize data with random values
    for (int i = 0; i < 1000; ++i) data[i] = rand() % 10000;

    double start = omp_get_wtime();
    parallelMergeSort(data, 0, data.size() - 1);
    double end = omp_get_wtime();

    std::cout << "Pipelined Parallel Merge Sort Time: " << (end - start) << " seconds\n";
    return 0;
} 





ASSIGNMENT 5
Q1.

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

// Statically defined global device arrays
__device__ float d_A[N];
__device__ float d_B[N];
__device__ float d_C[N];

// Kernel function for vector addition
__global__ void vectorAdd() {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) {
        d_C[i] = d_A[i] + d_B[i];
    }
}

int main() {
    float h_A[N], h_B[N], h_C[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Copy data from host to device
    cudaMemcpyToSymbol(d_A, h_A, N * sizeof(float));
    cudaMemcpyToSymbol(d_B, h_B, N * sizeof(float));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Record start event
    cudaEventRecord(start, 0);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>();

    // Record stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // in milliseconds

    // Copy result from device to host
    cudaMemcpyFromSymbol(h_C, d_C, N * sizeof(float));

    // Verify result
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            printf("Error at index %d: %f + %f != %f\n", i, h_A[i], h_B[i], h_C[i]);
            return -1;
        }
    }

    printf("Vector addition successful.\n");
    printf("Kernel execution time: %f ms\n", elapsedTime);

    // Query device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Memory clock rate in KHz, bus width in bits
    int memClockRate = prop.memoryClockRate;
    int memBusWidth = prop.memoryBusWidth;

    // Calculate theoretical bandwidth in GB/s
    double bandwidth = 2.0 * memClockRate * (memBusWidth / 8.0) / 1e6;
    printf("Theoretical Memory Bandwidth: %.2f GB/s\n", bandwidth);

    // Calculate measured bandwidth
    float totalBytes = N * (2 + 1) * sizeof(float); // 3 * N * 4 bytes
    double elapsedTimeSec = elapsedTime / 1000.0; // Convert ms to seconds
    double measuredBW = totalBytes / (elapsedTimeSec * 1e9); // GB/s
    printf("Measured Memory Bandwidth: %.2f GB/s\n", measuredBW);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
