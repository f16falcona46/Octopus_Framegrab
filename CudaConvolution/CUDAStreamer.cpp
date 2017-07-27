#include "CUDAStreamer.h"
#include <stdexcept>
#include "cuda_runtime.h"

void CUDAStreamer::StreamFunc(CUDAStreamer* streamer)
{
	while (streamer->m_streaming) {
		if (streamer->m_cons_in->size() <= 0) continue;
		CUDAStreamer::Consumer_element_t* in_buf = streamer->m_cons_in->front();
		streamer->m_cons_in->pop_front();
		for (int i = 0; i < streamer->m_bufcount; ++i) {
			streamer->m_host_in_buf[i] = in_buf[i];
		}
		streamer->m_cons_out->push_back(in_buf);
		cudaMemcpy(streamer->m_device_in_buf, streamer->m_host_in_buf, streamer->m_in_bufsize, cudaMemcpyHostToDevice);
		cufftExecR2C(streamer->m_plan, streamer->m_device_in_buf, streamer->m_device_out_buf);
		if (streamer->m_prod_in->size() <= 0) continue;
		CUDAStreamer::Producer_element_t* out_buf = streamer->m_prod_in->front();
		cudaMemcpy(out_buf, streamer->m_device_out_buf, streamer->m_out_bufsize, cudaMemcpyDeviceToHost);
		streamer->m_prod_out->push_back(out_buf);
	}
}

CUDAStreamer::CUDAStreamer() : m_setup(false), m_streaming(false)
{
}


CUDAStreamer::~CUDAStreamer()
{
	StopStreaming();
	cufftDestroy(m_plan);
	delete[] m_host_in_buf;
	cudaFree(m_device_in_buf);
	cudaFree(m_device_out_buf);
}

void CUDAStreamer::Setup()
{
	m_in_bufsize = sizeof(Consumer_element_t) * m_bufcount;
	m_out_bufsize = sizeof(Producer_element_t) * m_bufcount;
	m_host_in_buf = new cufftReal[m_bufcount];
	if (cudaMalloc(&m_device_in_buf, m_in_bufsize) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA input buffer.");
	if (cudaMalloc(&m_device_out_buf, m_out_bufsize) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA input buffer.");
	cufftPlan1d(&m_plan, m_linewidth, CUFFT_R2C, m_bufcount / m_linewidth);
	m_setup = true;
}

void CUDAStreamer::StartStreaming()
{
	if (!m_setup) throw std::runtime_error("CUDAStreamer wasn't set up before calling StartStreaming().");
	m_streaming = true;
	m_streamthread = std::thread(StreamFunc, this);
}

void CUDAStreamer::StopStreaming()
{
	m_streaming = false;
	m_streamthread.join();
}
