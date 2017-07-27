#include <Windows.h>
#include <iostream>
#include <memory>
#include "cuda_runtime.h"
#include "cufft.h"

#include "BufferQueue.h"
#include <vector>
#include <future>
#include <deque>

void acquire();
void square(float* arr, int size);

void worker_f(BufferQueue<uint16_t*>* in, BufferQueue<uint16_t*>* out)
{
	while (1) {
		std::deque<uint16_t*> toprocess;
		while (in->size() > 0) {
			if (in->front() == nullptr) {
				return;
			}
			toprocess.emplace_back(in->front());
			in->pop_front();
		}
		while (toprocess.size() > 0) {
			uint16_t* arr = toprocess.front();
			toprocess.pop_front();
			for (int i = 0; i < 2048; ++i) {
				++arr[i];
			}
			out->push_back(arr);
		}
	}
}

int main()
{
	BufferQueue<uint16_t*> to_worker;
	BufferQueue<uint16_t*> from_worker;
	std::mutex m;
	std::condition_variable cv;

	std::vector<std::unique_ptr<uint16_t[]>> buffers;
	for (int i = 0; i < 10; ++i) {
		buffers.emplace_back(new uint16_t[2048]);
		from_worker.push_back(buffers[i].get());
	}
	std::thread t(worker_f, &to_worker, &from_worker);
	
	for (int i = 0; i < 1000; ++i) {
		if (from_worker.size() > 0) {
			uint16_t* arr = from_worker.front();
			from_worker.pop_front();
			std::cout << arr[0] << '\n';
			to_worker.push_back(arr);
		}
	}
	to_worker.push_back(nullptr);
	for (int i = 0; i < 1000; ++i) {
		if (from_worker.size() > 0) {
			uint16_t* arr = from_worker.front();
			from_worker.pop_front();
			std::cout << arr[0] << '\n';
		}
	}

	/*
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
	*/
	MessageBox(NULL, "look", "", MB_OK);
}