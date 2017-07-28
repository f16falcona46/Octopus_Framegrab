#include <Windows.h>
#include <iostream>
#include <memory>
#include "cuda_runtime.h"
#include "cufft.h"

#include "BufferQueue.h"
#include "FrameGrabStreamer.h"
#include "CUDAStreamer.h"
#include "SaveStreamer.h"
#include "DisplayStreamer.h"
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
	DisplayStreamer ds;
	ds.SetLineLength(5);
	ds.Setup();
	ds.StartStreaming();
	MessageBox(NULL, "wait", "", MB_OK);
	ds.StopStreaming();
	MessageBox(NULL, "wait2", "", MB_OK);
	try {
		FrameGrabStreamer fgs;
		fgs.SetServerId(1);
		fgs.SetConfigFile("C:/Users/SD-OCT/Desktop/TargetCam.ccf");
		fgs.Setup();

		BufferQueue<FrameGrabStreamer::Producer_element_t*> fg_to_cuda;
		BufferQueue<FrameGrabStreamer::Producer_element_t*> cuda_to_fg;
		std::vector<std::unique_ptr<FrameGrabStreamer::Producer_element_t[]>> fgs_cuda_buffers;
		for (int i = 0; i < 256; ++i) {
			fgs_cuda_buffers.emplace_back(new FrameGrabStreamer::Producer_element_t[fgs.GetFrameHeight() * fgs.GetFrameWidth()]);
			cuda_to_fg.push_back(fgs_cuda_buffers[i].get());
		}
		BufferQueue<CUDAStreamer::Producer_element_t*> cuda_to_ssd;
		BufferQueue<CUDAStreamer::Producer_element_t*> ssd_to_cuda;
		std::vector<std::unique_ptr<CUDAStreamer::Producer_element_t[]>> cuda_ssd_buffers;
		for (int i = 0; i < 1024; ++i) {
			cuda_ssd_buffers.emplace_back(new CUDAStreamer::Producer_element_t[fgs.GetFrameHeight() * fgs.GetFrameWidth()]);
			ssd_to_cuda.push_back(cuda_ssd_buffers[i].get());
		}
		std::cout << sizeof(CUDAStreamer::Producer_element_t) * fgs.GetFrameHeight() * fgs.GetFrameWidth() << '\n';

		fgs.SetProducerInputQueue(&cuda_to_fg);
		fgs.SetProducerOutputQueue(&fg_to_cuda);

		CUDAStreamer cs;
		cs.SetConsumerInputQueue(&fg_to_cuda);
		cs.SetConsumerOutputQueue(&cuda_to_fg);
		cs.SetProducerInputQueue(&ssd_to_cuda);
		cs.SetProducerOutputQueue(&cuda_to_ssd);
		cs.SetLineWidth(fgs.GetFrameWidth());
		cs.SetBufferCount(fgs.GetFrameHeight() * fgs.GetFrameWidth());
		cs.Setup();

		SaveStreamer ss;
		ss.SetFilename("D:/out_stream.bin");
		ss.SetBufferCount(fgs.GetFrameHeight() * fgs.GetFrameWidth());
		ss.SetConsumerInputQueue(&cuda_to_ssd);
		ss.SetConsumerOutputQueue(&ssd_to_cuda);
		ss.Setup();

		fgs.StartStreaming();
		cs.StartStreaming();
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
		//while (1) {
		//	std::cout << cuda_to_fg.size() << ' ' << ssd_to_cuda.size() << '\n';
		//}
		fgs.StopStreaming();
		cs.StopStreaming();
		ss.StopStreaming();
		MessageBox(NULL, "Wait...", "", MB_OK);
	}
	catch (const std::exception& e) {
		std::cout << e.what() << "\n";
	}
	catch (...) {
		std::cout << "Something bad happened.\n";
	}
	MessageBox(NULL, "Wait more...", "", MB_OK);
}