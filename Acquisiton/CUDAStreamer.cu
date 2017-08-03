#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include "CUDAStreamer.h"

#ifdef NDEBUG
#define gpuErrchk(ans) ans
#else
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		//if (abort) exit(code);
	}
}
#endif

__global__ void Interpolate(const CUDAStreamer::Consumer_element_t* in, const CUDAStreamer::Consumer_element_t* dc, cufftComplex* out, const int* indexes, const float* fractions, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		out[idx].x = in[indexes[idx]] * (1.0f - fractions[idx]) + in[indexes[idx] + 1] * fractions[idx];
		out[idx].x -= dc[indexes[idx]] * (1.0f - fractions[idx]) + dc[indexes[idx] + 1] * fractions[idx];
		//out[idx].x = in[idx];
		out[idx].y = 0;
	}
}

__global__ void NormAndCopy(const cufftComplex* in, CUDAStreamer::Producer_element_t* out, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)	out[idx] = logf(sqrtf(in[idx].x * in[idx].x + in[idx].y * in[idx].y) + 10);
}

void CUDAStreamer::DoFFT(CUDAStreamer* streamer)
{
	Interpolate<<<(streamer->m_bufcount + 32 - 1) / 32, 32 >>>(streamer->m_device_in_buf, streamer->m_device_dc_buf, streamer->m_device_conv_in_buf, streamer->m_device_lerp_index, streamer->m_device_lerp_fraction, streamer->m_bufcount);
	if (cufftExecC2C(streamer->m_plan, streamer->m_device_conv_in_buf, streamer->m_device_out_buf, CUFFT_FORWARD) != CUFFT_SUCCESS) throw std::runtime_error("Failed to perform FFT.");
	//NormAndCopy <<<(streamer->m_bufcount + 32 - 1) / 32, 32 >>> (streamer->m_device_conv_in_buf, streamer->m_device_norm_out_buf, streamer->m_bufcount);
	NormAndCopy <<<(streamer->m_bufcount + 32 - 1) / 32, 32 >>> (streamer->m_device_out_buf, streamer->m_device_norm_out_buf, streamer->m_bufcount);
}
