#include <cstdlib> // malloc(), free()
#include <ctime> // time(), clock()
#include <cmath> // sqrt()
#include <iostream> // cout, stream
#include "common.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>

const int ITERS = 1000;
int size;

/* Entry point for the program. Allocates space for two matrices,
calls a function to multiply them, and displays the results. */
int main()
{
	cudaSetDeviceFlags(cudaDeviceMapHost);

	printf("please enter length of the vector ");
	scanf("%d",&size);

	// Number of bytes in the matrix.
	int bytes = size * sizeof(float);

	// Timing data
	float *a, *b,*ccpu,*cgpu,tcpu, tgpu;
	clock_t start, end;

	// Allocate the three arrays of SIZE x SIZE floats.
	// The element i,j is represented by index (i*SIZE + j)
	//float* a = new float[size];
	//float* b = new float[size];
	//float* ccpu = new float[size];
	//float* cgpu = new float[size];

	cudaHostAlloc((void**) &a, bytes, cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**) &b, bytes, cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**) &cgpu, bytes, cudaHostAllocWriteCombined | cudaHostAllocMapped);
	ccpu = (float*)malloc(bytes);

	// Initialize M and N to random integers
	for (int i = 0; i < size; i++) {
	a[i] = (float)(rand() % 10);
	b[i] = (float)(rand() % 10);
	}
////////////////////////////////////////////////////////////////////////////////////////////////////
	// add the two matrices on the host
	start = clock();
	for (int i = 0; i < ITERS; i++) {
	addVectorCPU( a, b, ccpu, size );
	}
	end = clock();

	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

	// Display the results
	std::cout << "\nCPU Addition took " << tcpu << " ms" << std::endl;
	
	// add the two matrices on the device
	// Perform one warm-up pass and validate
	bool success = addVectorGPU( a, b, cgpu, size );
	if (!success) {
	std::cout << "\n * Device error! * \n" << std::endl;
	return 1;
	}

	// And now time it
	start = clock();
	for (int i = 0; i < ITERS; i++) {
	addVectorGPU( a, b, cgpu, size );
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

	// Display the results
	std::cout << "GPU Addition took " << tgpu << " ms" << std::endl;
	std::cout << "Addition speedup = " << tcpu/tgpu <<std::endl;


	// Compare the results for correctness
	float sum = 0, delta = 0;
	for (int i = 0; i < size; i++) {
	delta += (ccpu[i] - cgpu[i]) * (ccpu[i] - cgpu[i]);
	sum += (ccpu[i] * cgpu[i]);
	}
	float L2norm = sqrt(delta / sum);
	std::cout << "Addition error = " << L2norm << "\n" << std::endl;


///////////////////////////////////////////////////////////////////////////////////////////
	// subtract the two matrices on the host
	start = clock();
	for (int i = 0; i < ITERS; i++) {
	subtractVectorCPU( a, b, ccpu, size );
	}
	end = clock();

	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

	// Display the results
	std::cout << "CPU Subtraction took " << tcpu << " ms" << std::endl;
	
	// add the two matrices on the device
	// Perform one warm-up pass and validate
	bool success1 = subtractVectorGPU( a, b, cgpu, size );
	if (!success1) {
	std::cout << "\n * Device error! * \n" << std::endl;
	return 1;
	}
	

	// And now time it
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		
	subtractVectorGPU( a, b, cgpu, size );
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

	// Display the results
	std::cout << "GPU Subtraction took " << tgpu << " ms" << std::endl;
	std::cout << " Subtraction speedup = " << tcpu/tgpu <<std::endl;


	// Compare the results for correctness
	float sum1 = 0, delta1 = 0;
	for (int i = 0; i < size; i++) {
	delta1 += (ccpu[i] - cgpu[i]) * (ccpu[i] - cgpu[i]);
	sum1 += (ccpu[i] * cgpu[i]);
	}
	float L2norm1 = sqrt(delta1 / sum1);
	std::cout << "Subtraction error =  " << L2norm1 << "\n" << std::endl;

//////////////////////////////////////////////////////////////////////////////////////////

	// scale matrices on the host
	float scaleFactor = 4;
	start = clock();
	for (int i = 0; i < ITERS; i++) {
	scaleVectorCPU( a, ccpu, scaleFactor, size );
	}
	end = clock();

	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

	// Display the results
	std::cout << "CPU Scale took " << tcpu << " ms" << std::endl;
	
	// add the two matrices on the device
	// Perform one warm-up pass and validate
	bool success2 = scaleVectorGPU( a, cgpu, scaleFactor, size);
	if (!success2) {
	std::cout << "\n * Device error! * \n" << std::endl;
	return 1;
	}

	// And now time it
	start = clock();
	for (int i = 0; i < ITERS; i++) {
	scaleVectorGPU( a, cgpu, scaleFactor, size );
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

	// Display the results
	std::cout << "GPU Scale took " << tgpu << " ms" << std::endl;
	std::cout << "Scale speedup =  " << tcpu/tgpu <<std::endl;


	// Compare the results for correctness
	float sum2 = 0, delta2 = 0;
	for (int i = 0; i < size; i++) {
	delta2 += (ccpu[i] - cgpu[i]) * (ccpu[i] - cgpu[i]);
	sum2 += (ccpu[i] * cgpu[i]);
	}
	float L2norm2 = sqrt(delta2 / sum2);
	std::cout << "Scale error =  " << L2norm2 << "\n" << std::endl;
/////////////////////////////////////////////////////////////////////////////////////////

	_sleep(1000000);
	getchar();

	// Release the matrices
	//delete[] a; delete[] b; delete[] ccpu; delete[] cgpu;
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(cgpu);
	delete[] ccpu;
	// Success
	return 0;
}