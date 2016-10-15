#include "common.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <math.h>

// GPU Kernel
__global__ void scaleVectorGPUKernel( float* ad, float* cd, float scaleFactor, int size ){

	// Retrieve our coordinates in the block
	int tx = blockIdx.x * blockDim.x + threadIdx.x;


	// Perform
	if(tx<size){
	cd[tx]=ad[tx] * scaleFactor;
	}

}

bool scaleVectorGPU( float* a, float* c, float scaleFactor, int size ){

	// Error return value
	cudaError_t status;

	// Number of bytes in the matrix.
	int bytes = size * sizeof(float);

	// Pointers to the device arrays
	float *ad,*cd;

	// Allocate memory on the device to store each matrix
	cudaMalloc((void**) &ad, bytes);
	//cudaMalloc((void**) &bd, bytes);
	cudaMalloc((void**) &cd, bytes);

	// Copy the host input data to the device
	cudaMemcpy(ad, a, bytes, cudaMemcpyHostToDevice);
	//cudaMemcpy(bd, b, bytes, cudaMemcpyHostToDevice);

	
	
	// Specify the size of the grid and the size of the block
	float dimBlock= 1024; // Matrix is contained in a block
	float x = (size/dimBlock);
	int dimGrid = (int)ceil(x);// Only using a single grid element today

	// Launch the kernel on a size-by-size block of threads
	scaleVectorGPUKernel<<<dimGrid, dimBlock>>>(ad, cd, scaleFactor, size);
	// Wait for completion
	cudaThreadSynchronize();
	// Retrieve the result matrix
	cudaMemcpy(c, cd, bytes, cudaMemcpyDeviceToHost);

		// Check for errors
	status = cudaGetLastError();
	if (status != cudaSuccess) {
	std::cout << "Kernel failed: " <<
	cudaGetErrorString(status) << std::endl;
	cudaFree(ad);
	//cudaFree(bd);
	cudaFree(cd);
	return false;
	}

	


	// Free device memory
	cudaFree(ad);
	//cudaFree(bd);
	cudaFree(cd);


	// Success
	return true;
}