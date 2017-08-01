#include <Windows.h>
#include <iostream>
#include <memory>
#include "cuda_runtime.h"
#include "cufft.h"

#include "BufferQueue.h"
#include "LoadStreamer.h"
#include "FrameGrabStreamer.h"
#include "CUDAStreamer.h"
#include "SaveStreamer.h"
#include "DisplayStreamer.h"
#include <vector>
#include <future>
#include <deque>

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
		fgs.SetFrameHeight(1000);
		fgs.SetInputFile("C:/Users/SD-OCT/Desktop/OCT/Data/Tumor_NewCCD_2017-03-08_00000.oct");
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
		MessageBox(NULL, "look", "", MB_OK);
		
		/*
		while (1) {
			std::cout << fg_to_cuda.size() << '\t' << cuda_to_fg.size() << '\t' << cuda_to_ssd.size() << '\t' << ssd_to_cuda.size() << '\n';
		}
		*/
		
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