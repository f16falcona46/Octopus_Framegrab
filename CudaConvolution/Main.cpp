#include <Windows.h>
#include <iostream>
#include <memory>
#include "cuda_runtime.h"
#include "cufft.h"

#include "BufferQueue.h"
#include "FrameGrabStreamer.h"
#include "SaveStreamer.h"
#include <vector>
#include <future>
#include <deque>

void acquire();
void square(float* arr, int size);

/*
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
*/

int main()
{
	BufferQueue<uint16_t*> to_worker;
	BufferQueue<uint16_t*> from_worker;

	FrameGrabStreamer fgs;
	fgs.SetServerId(1);
	fgs.SetConfigFile("C:/Users/SD-OCT/Desktop/TargetCam.ccf");
	fgs.Setup();

	std::vector<std::unique_ptr<FrameGrabStreamer::Producer_element_t[]>> buffers;
	for (int i = 0; i < 256; ++i) {
		buffers.emplace_back(new FrameGrabStreamer::Producer_element_t[fgs.GetFrameHeight() * fgs.GetFrameWidth()]);
		from_worker.push_back(buffers[i].get());
	}

	fgs.SetProducerInputQueue(&from_worker);
	fgs.SetProducerOutputQueue(&to_worker);
	
	SaveStreamer ss;
	ss.SetFilename("D:/out_stream.bin");
	ss.SetBufferSize(sizeof(FrameGrabStreamer::Producer_element_t) * fgs.GetFrameHeight() * fgs.GetFrameWidth());
	ss.SetConsumerInputQueue(&to_worker);
	ss.SetConsumerOutputQueue(&from_worker);
	ss.Setup();

	fgs.StartStreaming();
	ss.StartStreaming();
	/*while (1) {
		if (from_worker.size() > 0) {
			uint16_t* buf = from_worker.front();
			from_worker.pop_front();
			to_worker.push_back(buf);
		}
	}*/
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
	fgs.StopStreaming();
	ss.StopStreaming();
}