#include <Windows.h>
#include <iostream>
#include <memory>
#include "cuda_runtime.h"
#include "cufft.h"

void acquire();
void square(float* arr, int size);

int main()
{
	acquire();
	std::unique_ptr<float[]> arr(new float[1024]);
	for (int i = 0; i < 1024; ++i) {
		arr[i] = i;
	}
	square(arr.get(), 1024);
	for (int i = 0; i < 20; ++i) {
		std::cout << i << ": " << arr[i] << '\n';
	}

	cufftComplex fc_arr[4];
	fc_arr[0].x = 1;
	fc_arr[0].y = 0;
	fc_arr[1].x = 0;
	fc_arr[1].y = 0;
	fc_arr[2].x = -1;
	fc_arr[2].y = 0;
	fc_arr[3].x = 0;
	fc_arr[3].y = 0;
	cufftComplex* df_arr;
	cudaMalloc(&df_arr, 4 * sizeof(cufftComplex));
	cudaMemcpy(df_arr, fc_arr, 4 * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	cufftHandle plan;
	cufftPlan1d(&plan, 4, CUFFT_C2C, 1);
	cufftExecC2C(plan, df_arr, df_arr, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	cudaMemcpy(fc_arr, df_arr, 4 * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	cufftDestroy(plan);
	cudaFree(df_arr);
	for (int i = 0; i < 4; ++i) {
		std::cout << fc_arr[i].x << ',' << fc_arr[i].y << '\n';
	}
	MessageBox(NULL, "look", "", MB_OK);
}