#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include "CUDAStreamer.h"

__global__ void Interpolate(const uint16_t* in, float* out, const int* indexes, const float* fractions)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	out[idx] = in[indexes[idx]] * (1.0f - fractions[idx]) + in[indexes[idx] + 1] * fractions[idx];
}

/*
__global__ void ToRealAndCopy(const CUDAStreamer::Consumer_element_t* in, cufftReal* out)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	out[idx] = in[idx];
}
*/

__global__ void NormAndCopy(const cufftComplex* in, CUDAStreamer::Producer_element_t* out)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	out[idx] = sqrt(in[idx].x * in[idx].x + in[idx].y * in[idx].y);
}

void CUDAStreamer::DoFFT(CUDAStreamer* streamer)
{
	Interpolate<<<1, streamer->m_bufcount >>>(streamer->m_device_in_buf, streamer->m_device_conv_in_buf, streamer->m_device_lerp_index, streamer->m_device_lerp_fraction);
	cufftExecR2C(streamer->m_plan, streamer->m_device_conv_in_buf, streamer->m_device_out_buf);
	NormAndCopy <<<1, streamer->m_bufcount >>> (streamer->m_device_out_buf, streamer->m_device_norm_out_buf);

	cudaDeviceSynchronize();
}
