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
		if (abort) exit(code);
	}
}
#endif

/*
\brief Kernel that conducts linear interpolation on one value

In addition to interpolation, this also casts the uint16_t provided by in to cufftComplex
required for the FFT.

\param in	- the values to be interpolated
\param dc	- an A-line of DC to be subtracted from the data
\param out	- where the result should be stored
\param indexes	- the index of the lower sample that the target sample is bracketed between
\param fractions	- the part of the way that the target sample is from the lower sample
\param size	- the total number of pixels
\param linewidth	- the length of one A-line, in pixels
*/
__global__ void Interpolate(const CUDAStreamer::Consumer_element_t* in, const float* dc, cufftComplex* out, const int* indexes, const float* fractions, int size, int linewidth)
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

/*
\brief Kernel that takes the norm of an FFT result value and copies it to a float buffer

This function also takes the log (base e) of the data.

\param in	- FFT result values to norm
\param out	- place to store the results
\param size	- total number of pixels per frame
\param linewidth	- number of pixels per A-line
*/
__global__ void NormAndCopy(const cufftComplex* in, CUDAStreamer::Producer_element_t* out, int size, int linewidth)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)	out[idx] = logf(sqrtf(in[idx].x * in[idx].x + in[idx].y * in[idx].y) + 10);
}

/*
\brief Given the maximum and minimum pixels, adjusts the contrast of the final image.

This function is not used yet, since using thrust to find the maximum and minimum pixels does not work.

\param in	- data to adjust
\param out	- intensity-adjusted value
\param min	- minimum pixel intensity value
\param max	- maximum pixel intensity value
\param size	- total number of pixels in the frame
*/
__global__ void AdjustContrast(const CUDAStreamer::Producer_element_t* in, CUDAStreamer::Producer_element_t* out, CUDAStreamer::Producer_element_t min, CUDAStreamer::Producer_element_t max, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		out[idx] = (in[idx] - min) / (max - min);
	}
}

/*
\brief Kernel that copies a value to a CUDA buffer

\param buf	- buffer to fill
\param size	- number of values to fill
\param value	- value to fill the buffer with
*/
__global__ void Fill(cufftComplex* buf, int size, cufftComplex value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) buf[idx] = value;
}

/*
\brief Fills a CUDA buffer with a certain value.

\param buf	- buffer to fill
\param size	- number of values to fill
\param value	- value to fill the buffer with
*/
void CUDAStreamer::FillBuffer(cufftComplex* buf, int size, cufftComplex value)
{
	Fill <<<(size + 32 - 1) / 32, 32 >>> (buf, size, value);
}

/*
\brief Processes raw OCT data into images

This function first transforms the raw OCT wavelength data into wavenumber data, using
linear interpolation. Next, an FFT is performed, and then the FFT output data is normed
and copied to a result buffer.

\param streamer	- CUDAStreamer that holds the buffers and data to perform the FFT on
*/
void CUDAStreamer::DoFFT(CUDAStreamer* streamer)
{
	Interpolate<<<(streamer->m_bufcount + 32 - 1) / 32, 32 >>>(
		streamer->m_device_in_buf, streamer->m_device_dc_buf,
		streamer->m_device_conv_in_buf, streamer->m_device_lerp_index,
		streamer->m_device_lerp_fraction, streamer->m_bufcount, streamer->m_linewidth);
	if (cufftExecC2C(streamer->m_plan, streamer->m_device_conv_in_buf, streamer->m_device_out_buf, CUFFT_FORWARD) != CUFFT_SUCCESS) throw std::runtime_error("Failed to perform FFT.");
	//NormAndCopy << <(streamer->m_bufcount + 32 - 1) / 32, 32 >> > (streamer->m_device_out_buf, streamer->m_device_norm_out_buf, streamer->m_bufcount, streamer->m_linewidth);
	NormAndCopy <<<(streamer->m_bufcount + 32 - 1) / 32, 32 >>> (streamer->m_device_out_buf, streamer->m_device_contrast_out_buf, streamer->m_bufcount, streamer->m_linewidth);
	//thrust::device_ptr<CUDAStreamer::Producer_element_t> device_norm_out_buf = thrust::device_pointer_cast(streamer->m_device_norm_out_buf);
	//CUDAStreamer::Producer_element_t min_pix = thrust::reduce(device_norm_out_buf, device_norm_out_buf + streamer->m_bufcount, 65536.0f, thrust::min<float>);
	//CUDAStreamer::Producer_element_t max_pix = thrust::reduce(device_norm_out_buf, device_norm_out_buf + streamer->m_bufcount, -65536.0f, thrust::max<float>);
	//AdjustContrast<<<(streamer->m_bufcount + 32 - 1) / 32, 32>>>(streamer->m_device_norm_out_buf, streamer->m_device_contrast_out_buf, min_pix, max_pix, streamer->m_bufcount);
}
