#include <Windows.h>
#include <iostream>
#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>

#include "BufferQueue.h"
#include "LoadStreamer.h"
#include "FrameGrabStreamer.h"
#include "CUDAStreamer.h"
#include "SaveStreamer.h"
#include "ShortSaveStreamer.h"
#include "DisplayStreamer.h"

template <typename T>
void Transpose(T* src, T* dst, const int N, const int M) {
	for (int n = 0; n<N*M; n++) {
		int i = n / N;
		int j = n%N;
		dst[n] = src[M*j + i];
	}
}

int main()
{
	try {
		/*
		FrameGrabStreamer fgs;
		fgs.SetServerId(1);
		fgs.SetConfigFile("C:/Users/SD-OCT/Desktop/TargetCam.ccf");
		fgs.Setup();
		*/
		LoadStreamer fgs;
		fgs.SetFrameWidth(2048);
		fgs.SetFrameHeight(2048);
		fgs.SetInputFile("C:/Users/SD-OCT/Desktop/OCT/Data/Tumor_NewCCD_2017-03-08_00000.oct.raw");
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
		std::vector<CUDAStreamer::Consumer_element_t> dc_buf(fgs.GetFrameHeight() * fgs.GetFrameWidth());
		std::vector<CUDAStreamer::Consumer_element_t> dc_buf_transposed(fgs.GetFrameHeight() * fgs.GetFrameWidth());
		std::ifstream dc_file("C:/Users/SD-OCT/Desktop/OCT/Data/dc.raw", std::ios::in | std::ios::binary);
		dc_file.read((char*)dc_buf.data(), dc_buf.size() * sizeof(CUDAStreamer::Consumer_element_t));
		Transpose(dc_buf.data(), dc_buf_transposed.data(), fgs.GetFrameWidth(), fgs.GetFrameHeight());
		cs.CopyDCBuffer(dc_buf_transposed.data());

		SaveStreamer ss;
		ss.SetFilename("D:/out_stream.bin");
		ss.SetBufferCount(fgs.GetFrameHeight() * fgs.GetFrameWidth());
		ss.SetConsumerInputQueue(&cuda_to_ssd);
		ss.SetConsumerOutputQueue(&ssd_to_cuda);
		ss.Setup();

		fgs.StartStreaming();
		cs.StartStreaming();
		ss.StartStreaming();
		MessageBox(NULL, "look", "", MB_OK);
		
		fgs.StopStreaming();
		cs.StopStreaming();
		ss.StopStreaming();
		MessageBox(NULL, "Wait...", "", MB_OK);
	}
	catch (const std::exception& e) {
		std::cout << "Exception: " << e.what() << "\n";
		throw e;
	}
	catch (...) {
		std::cout << "Something bad happened.\n";
	}
	MessageBox(NULL, "Wait more...", "", MB_OK);
}