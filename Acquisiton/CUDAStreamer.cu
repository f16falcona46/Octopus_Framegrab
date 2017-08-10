#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <device_launch_parameters.h>
#include <cufft.h>
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

__global__ void Interpolate(const CUDAStreamer::Consumer_element_t* in, const float* dc, cufftComplex* out, const int* indexes, const float* fractions, int linewidth, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int interp_idx = idx % linewidth;
	int offset = idx - interp_idx;
	if (idx < size) {
		out[idx].x = in[indexes[interp_idx] + offset] * (1.0f - fractions[interp_idx]) + in[indexes[interp_idx] + offset - 1] * fractions[interp_idx];
		out[idx].x -= dc[indexes[interp_idx]] * (1.0f - fractions[interp_idx]) + dc[indexes[interp_idx] - 1] * fractions[interp_idx];
		out[idx].y = 0;
	}
}

__global__ void NormAndCopy(const cufftComplex* in, CUDAStreamer::Producer_element_t* out, int size, int linewidth)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)	out[idx] = logf(sqrtf(in[idx].x * in[idx].x + in[idx].y * in[idx].y) + 10);
}

__global__ void AdjustContrast(const CUDAStreamer::Producer_element_t* in, CUDAStreamer::Producer_element_t* out, CUDAStreamer::Producer_element_t min, CUDAStreamer::Producer_element_t max, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		out[idx] = (in[idx] - min) / (max - min);
	}
}

__global__ void Fill(cufftComplex* buf, int size, cufftComplex value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = 0;
	if (idx < size) buf[idx] = value;
}

void CUDAStreamer::FillBuffer(cufftComplex* buf, int size, cufftComplex value)
{
	Fill <<<(size + 32 - 1) / 32, 32 >>> (buf, size, value);
}

void CUDAStreamer::DoFFT(CUDAStreamer* streamer)
{
	Interpolate<<<(streamer->m_bufcount + 32 - 1) / 32, 32 >>>(streamer->m_device_in_buf, streamer->m_device_dc_buf, streamer->m_device_conv_in_buf, streamer->m_device_lerp_index, streamer->m_device_lerp_fraction, streamer->m_linewidth, streamer->m_bufcount);
	if (cufftExecC2C(streamer->m_plan, streamer->m_device_conv_in_buf, streamer->m_device_out_buf, CUFFT_FORWARD) != CUFFT_SUCCESS) throw std::runtime_error("Failed to perform FFT.");
	//NormAndCopy << <(streamer->m_bufcount + 32 - 1) / 32, 32 >> > (streamer->m_device_out_buf, streamer->m_device_norm_out_buf, streamer->m_bufcount, streamer->m_linewidth);
	NormAndCopy <<<(streamer->m_bufcount + 32 - 1) / 32, 32 >>> (streamer->m_device_out_buf, streamer->m_device_contrast_out_buf, streamer->m_bufcount, streamer->m_linewidth);
	//thrust::device_ptr<CUDAStreamer::Producer_element_t> device_norm_out_buf = thrust::device_pointer_cast(streamer->m_device_norm_out_buf);
	//CUDAStreamer::Producer_element_t min_pix = thrust::reduce(device_norm_out_buf, device_norm_out_buf + streamer->m_bufcount, 65536.0f, thrust::min<float>);
	//CUDAStreamer::Producer_element_t max_pix = thrust::reduce(device_norm_out_buf, device_norm_out_buf + streamer->m_bufcount, -65536.0f, thrust::max<float>);
	//AdjustContrast<<<(streamer->m_bufcount + 32 - 1) / 32, 32>>>(streamer->m_device_norm_out_buf, streamer->m_device_contrast_out_buf, min_pix, max_pix, streamer->m_bufcount);
}
