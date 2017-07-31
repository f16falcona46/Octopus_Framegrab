#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void add1(float* arr)
{
	int t_id = threadIdx.x;
	arr[t_id] = arr[t_id] * arr[t_id];
}

void square(float* arr, int size)
{
	float* dev_arr;
	cudaMalloc(&dev_arr, size * sizeof(float));
	cudaMemcpy(dev_arr, arr, size * sizeof(float), cudaMemcpyHostToDevice);
	add1 <<<1, size >>> (dev_arr);
	cudaMemcpy(arr, dev_arr, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(dev_arr);
}