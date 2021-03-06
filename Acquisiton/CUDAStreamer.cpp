#define _USE_MATH_DEFINES
#include <cmath>
#include "CUDAStreamer.h"
#include <stdexcept>
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>

#include "cuda_runtime.h"

double CalibrationFunction(double pixel_idx)
{
	const float c0 = 7.35122E+02;
	const float c1 = 9.91797E-02;
	const float c2 = -4.86383E-06;
	const float c3 = -1.87312E-10;
	return c0 + pixel_idx * (c1 + pixel_idx * (c2 + pixel_idx * c3));
}

void CUDAStreamer::DestroyBuffers()
{
	cufftDestroy(m_plan);
	cudaFree(m_device_in_buf);
	cudaFree(m_device_conv_in_buf);
	cudaFree(m_device_out_buf);
	cudaFree(m_device_norm_out_buf);
	cudaFree(m_device_contrast_out_buf);
	cudaFree(m_device_lerp_index);
	cudaFree(m_device_lerp_fraction);
	cudaFree(m_device_dc_buf);
	m_buffers_allocated = false;
}

void CUDAStreamer::StreamFunc(CUDAStreamer* streamer)
{
	while (streamer->m_streaming) {
		if (streamer->m_cons_in->size() > 0 && streamer->m_prod_in->size() > 0) {	//check if the input bufferqueues have buffers to use
			CUDAStreamer::Consumer_element_t* in_buf = streamer->m_cons_in->front();	//get the buffer at the front of the queue
			streamer->m_cons_in->pop_front();	//remove the buffer from the input queue
			cudaMemcpy(streamer->m_device_in_buf, in_buf, streamer->m_in_bufsize, cudaMemcpyHostToDevice);	//copy data to CUDA input array
			streamer->m_cons_out->push_back(in_buf);	//return buffer to end of output queue

			DoFFT(streamer);

			CUDAStreamer::Producer_element_t* out_buf = streamer->m_prod_in->front();
			streamer->m_prod_in->pop_front();
			cudaMemcpy(out_buf, streamer->m_device_contrast_out_buf, streamer->m_out_bufsize, cudaMemcpyDeviceToHost);
			streamer->m_prod_out->push_back(out_buf);
		}
	}
}

CUDAStreamer::CUDAStreamer() : m_setup(false), m_streaming(false), m_buffers_allocated(false)
{
}


CUDAStreamer::~CUDAStreamer()
{
	StopStreaming();
	if (m_buffers_allocated) DestroyBuffers();
}

void CUDAStreamer::Setup()
{
	if (m_buffers_allocated) DestroyBuffers();
	/*
	generate some lookup tables to help linear interpolation
	indexes is the index of the lower of the two samples
	fractions is how far the point is from the lower sample to the upper one

	for example:
	A.....b.C
	where A is index 3 in wavenumber, C is index 4 in wavenumber, and point b
	(which is index 6 in the linear_wavenumber vector) 3/4ths of the way to C
	between them is	the point to be interpolated

	then index[6] = 3
	and fraction[6] = 0.75
	*/

	//generate original sample X values
	std::vector<float> wavenumber;
	for (int i = 1; i <= m_linewidth; ++i) {
		wavenumber.emplace_back(2.0f * M_PI / CalibrationFunction(i));
	}
	//generate linear resample X values
	std::vector<float> linear_wavenumber;
	for (int i = 0; i < m_linewidth; ++i) {
		linear_wavenumber.emplace_back(wavenumber[wavenumber.size() - 1] + (wavenumber[0] - wavenumber[wavenumber.size() - 1]) / (m_linewidth - 1) * i);
	}
	std::vector<int> indexes;
	std::vector<float> fractions;
	int wavenumber_idx = m_linewidth - 1;
	for (int i = 0; i < m_linewidth; ++i) {
		while (1) {
			//either we're at an end of the array, or the linear resample X value in question is bracketed between two sample X values
			if (wavenumber_idx - 1 <= 0 || ((wavenumber[wavenumber_idx] <= linear_wavenumber[i] || wavenumber_idx == m_linewidth - 1) && wavenumber[wavenumber_idx - 1] >= linear_wavenumber[i])) {
				indexes.emplace_back(wavenumber_idx);
				fractions.emplace_back((linear_wavenumber[i] - wavenumber[wavenumber_idx]) / (wavenumber[wavenumber_idx - 1] - wavenumber[wavenumber_idx]));
				break;
			}
			//we're not, so try the next wavenumber X value
			else {
				--wavenumber_idx;
			}
		}
	}
	//allocate CUDA interpolation tables
	if (cudaMalloc(&m_device_lerp_index, m_bufcount * sizeof(int)) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA lerp index buffer.");
	if (cudaMalloc(&m_device_lerp_fraction, m_bufcount * sizeof(float)) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA lerp fraction buffer.");
	//copy the interpolation tables to CUDA
	if (cudaMemcpy(m_device_lerp_index, indexes.data(), m_linewidth * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("Couldn't copy linear interpolation indexes.");
	if (cudaMemcpy(m_device_lerp_fraction, fractions.data(), m_linewidth * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("Couldn't copy linear interpolation fractions.");

	//allocate buffers
	m_in_bufsize = m_bufcount * sizeof(CUDAStreamer::Consumer_element_t);
	m_out_bufsize = m_bufcount * sizeof(CUDAStreamer::Producer_element_t);
	if (cudaMalloc(&m_device_in_buf, m_bufcount * sizeof(CUDAStreamer::Consumer_element_t)) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA input buffer.");
	if (cudaMalloc(&m_device_conv_in_buf, m_bufcount * 2 * sizeof(cufftComplex)) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA converted input buffer.");
	cufftComplex zero;
	zero.x = 0;
	zero.y = 0;
	FillBuffer(m_device_conv_in_buf, m_bufcount * 2, zero);
	if (cudaMalloc(&m_device_out_buf, m_bufcount * 2 * sizeof(cufftComplex)) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA output buffer.");
	if (cudaMalloc(&m_device_norm_out_buf, m_bufcount * sizeof(CUDAStreamer::Producer_element_t)) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA converted output buffer.");
	if (cudaMalloc(&m_device_contrast_out_buf, m_bufcount * sizeof(CUDAStreamer::Producer_element_t)) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA contrast-adjusted output buffer.");
	if (cudaMalloc(&m_device_dc_buf, m_bufcount * sizeof(CUDAStreamer::Consumer_element_t)) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA DC frame buffer.");
	if (cudaMemset(m_device_dc_buf, 0, m_linewidth * sizeof(float)) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA DC frame buffer.");

	//initialize a plan for cuFFT
	if (cufftPlan1d(&m_plan, m_linewidth, CUFFT_C2C, m_bufcount / m_linewidth) != CUFFT_SUCCESS) throw std::runtime_error("Couldn't create cufft plan.");

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
	if (m_streamthread.joinable()) {
		m_streamthread.join();
	}
}

void CUDAStreamer::CopyDCBuffer(Consumer_element_t * buf)
{
	std::vector<float> line(m_linewidth);
	for (int i = 0; i < m_bufcount; ++i) {
		line[i % m_linewidth] += buf[i];
	}
	for (int i = 0; i < m_linewidth; ++i) {
		line[i] /= m_linewidth;
	}
	cudaMemcpy(m_device_dc_buf, line.data(), m_linewidth * sizeof(float), cudaMemcpyHostToDevice);
}
