#include "common.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <math.h>

// GPU Kernel
__global__ void subtractVectorGPUKernel( float* ad, float* bd, float* cd, int size ){

	// Retrieve our coordinates in the block
	int tx = blockIdx.x * blockDim.x + threadIdx.x;


	// Perform
	if(tx<size){
	cd[tx]=ad[tx] - bd[tx];
	}

	
}

bool subtractVectorGPU( float* a, float* b, float* c, int size ){

	// Error return value
	cudaError_t status;

	// Number of bytes in the matrix.
	int bytes = size * sizeof(float);

	// Pointers to the device arrays
	float *ad, *bd, *cd;

	// Allocate memory on the device to store each matrix
	cudaHostGetDevicePointer( (void**)&ad, a, 0 );
	cudaHostGetDevicePointer( (void**)&bd, b, 0 );
	cudaHostGetDevicePointer( (void**)&cd, c, 0 );
	// Specify the size of the grid and the size of the block
	float dimBlock= 1024; 
	float x = (size/dimBlock);
	int dimGrid = (int)ceil(x);

	// Launch the kernel on a size-by-size block of threads
	subtractVectorGPUKernel<<<dimGrid, dimBlock>>>(ad, bd, cd, size);
	// Wait for completion
	cudaThreadSynchronize();

	// Check for errors
	status = cudaGetLastError();
	if (status != cudaSuccess) {
	std::cout << "Kernel failed: " <<
	cudaGetErrorString(status) << std::endl;
	return false;
	}

	// Success
	return true;
}