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

void CUDAStreamer::StreamFunc(CUDAStreamer* streamer)
{
	while (streamer->m_streaming) {
		if (streamer->m_cons_in->size() > 0 && streamer->m_prod_in->size() > 0) {
			CUDAStreamer::Consumer_element_t* in_buf = streamer->m_cons_in->front();
			streamer->m_cons_in->pop_front();
			streamer->m_cons_out->push_back(in_buf);
			cudaMemcpy(streamer->m_device_in_buf, in_buf, streamer->m_in_bufsize, cudaMemcpyHostToDevice);

			DoFFT(streamer);

			CUDAStreamer::Producer_element_t* out_buf = streamer->m_prod_in->front();
			streamer->m_prod_in->pop_front();
			cudaMemcpy(out_buf, streamer->m_device_norm_out_buf, streamer->m_out_bufsize, cudaMemcpyDeviceToHost);
			//std::cout << out_buf[streamer->m_bufcount - 1] << '\n';
			streamer->m_prod_out->push_back(out_buf);
		}
	}
}

CUDAStreamer::CUDAStreamer() : m_setup(false), m_streaming(false)
{
}


CUDAStreamer::~CUDAStreamer()
{
	StopStreaming();
	cufftDestroy(m_plan);
	cudaFree(m_device_in_buf);
	cudaFree(m_device_conv_in_buf);
	cudaFree(m_device_out_buf);
	cudaFree(m_device_norm_out_buf);
}

void CUDAStreamer::Setup()
{
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

	std::vector<float> wavenumber;
	for (int i = 1; i <= m_linewidth; ++i) {
		wavenumber.emplace_back(2.0f * M_PI / CalibrationFunction(i));
	}
	std::vector<float> linear_wavenumber;
	for (int i = 0; i < m_linewidth; ++i) {
		linear_wavenumber.emplace_back(wavenumber[wavenumber.size() - 1] + (wavenumber[0] - wavenumber[wavenumber.size() - 1]) / (m_linewidth - 1) * i);
	}
	std::reverse(wavenumber.begin(), wavenumber.end());
	std::vector<int> indexes;
	std::vector<float> fractions;
	int wavenumber_idx = 0;
	for (int i = 0; i < m_linewidth; ++i) {
		while (1) {
			if (wavenumber_idx + 2 >= m_linewidth || wavenumber[wavenumber_idx + 1] > linear_wavenumber[i]) {
				indexes.emplace_back(wavenumber_idx);
				fractions.emplace_back((linear_wavenumber[i] - wavenumber[wavenumber_idx]) / (wavenumber[wavenumber_idx + 1] - wavenumber[wavenumber_idx]));
				break;
			}
			else {
				++wavenumber_idx;
			}
		}
	}
	std::unique_ptr<int[]> indexes_todevice(new int[m_bufcount]);
	std::unique_ptr<float[]> fractions_todevice(new float[m_bufcount]);
	int* in_i = indexes_todevice.get();
	float* in_f = fractions_todevice.get();
	while (in_i < indexes_todevice.get() + m_bufcount) {
		std::copy(indexes.begin(), indexes.end(), in_i);
		for (int& i : indexes) {
			i += indexes.size();
		}
		std::copy(fractions.begin(), fractions.end(), in_f);
		in_i += m_linewidth;
		in_f += m_linewidth;
	}
	if (cudaMalloc(&m_device_lerp_index, m_bufcount * sizeof(int)) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA lerp index buffer.");
	if (cudaMalloc(&m_device_lerp_fraction, m_bufcount * sizeof(float)) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA lerp fraction buffer.");
	cudaMemcpy(m_device_lerp_index, indexes_todevice.get(), m_bufcount * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(m_device_lerp_fraction, fractions_todevice.get(), m_bufcount * sizeof(float), cudaMemcpyHostToDevice);

	//allocate buffers
	m_in_bufsize = m_bufcount * sizeof(CUDAStreamer::Consumer_element_t);
	m_out_bufsize = m_bufcount * sizeof(CUDAStreamer::Producer_element_t);
	if (cudaMalloc(&m_device_in_buf, m_bufcount * sizeof(CUDAStreamer::Consumer_element_t)) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA input buffer.");
	if (cudaMalloc(&m_device_conv_in_buf, m_bufcount * sizeof(cufftComplex)) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA converted input buffer.");
	if (cudaMalloc(&m_device_out_buf, m_bufcount * sizeof(cufftComplex)) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA output buffer.");
	if (cudaMalloc(&m_device_norm_out_buf, m_bufcount * sizeof(CUDAStreamer::Producer_element_t)) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA converted output buffer.");
	if (cudaMalloc(&m_device_dc_buf, m_bufcount * sizeof(CUDAStreamer::Consumer_element_t)) != cudaSuccess) throw std::runtime_error("Couldn't allocate CUDA DC frame buffer.");
	if (cudaMemset(m_device_dc_buf, 0, m_bufcount * sizeof(CUDAStreamer::Consumer_element_t)) != cudaSuccess) throw std::runtime_error("Couldn't clear CUDA DC frame buffer.");
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
	cudaMemcpy(m_device_dc_buf, buf, m_in_bufsize, cudaMemcpyHostToDevice);
}
