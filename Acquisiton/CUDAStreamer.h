#pragma once
#include <cstdint>
#include <thread>
#include "BufferQueue.h"
#include "cufft.h"

class CUDAStreamer
{
public:
	typedef uint16_t Consumer_element_t;
	typedef float Producer_element_t;
	typedef BufferQueue<Consumer_element_t*> Consumer_queue_t;
	typedef BufferQueue<Producer_element_t*> Producer_queue_t;
	CUDAStreamer();
	~CUDAStreamer();
	Consumer_queue_t* GetConsumerInputQueue() { return m_cons_in; }
	void SetConsumerInputQueue(Consumer_queue_t* in) { m_cons_in = in; }
	Consumer_queue_t* GetConsumerOutputQueue() { return m_cons_out; }
	void SetConsumerOutputQueue(Consumer_queue_t* out) { m_cons_out = out; }
	Producer_queue_t* GetProducerInputQueue() { return m_prod_in; }
	void SetProducerInputQueue(Producer_queue_t* in) { m_prod_in = in; }
	Producer_queue_t* GetProducerOutputQueue() { return m_prod_out; }
	void SetProducerOutputQueue(Producer_queue_t* out) { m_prod_out = out; }
	void Setup();
	void StartStreaming();
	void StopStreaming();
	void SetBufferCount(size_t bufcount) { m_bufcount = bufcount; }
	void SetLineWidth(size_t width) { m_linewidth = width; }
	void CopyDCBuffer(Consumer_element_t* buf);

private:
	Consumer_queue_t* m_cons_in;
	Consumer_queue_t* m_cons_out;
	Producer_queue_t* m_prod_in;
	Producer_queue_t* m_prod_out;
	std::thread m_streamthread;
	bool m_streaming;
	bool m_setup;
	cufftHandle m_plan;
	uint16_t* m_device_in_buf;
	cufftComplex* m_device_conv_in_buf;
	cufftComplex* m_device_out_buf;
	Producer_element_t* m_device_norm_out_buf;
	int* m_device_lerp_index;
	float* m_device_lerp_fraction;
	Consumer_element_t* m_device_dc_buf;
	size_t m_bufcount;
	size_t m_linewidth;
	size_t m_in_bufsize;
	size_t m_out_bufsize;

	static void CUDAStreamer::StreamFunc(CUDAStreamer* streamer);
	static void DoFFT(CUDAStreamer* streamer);
};

