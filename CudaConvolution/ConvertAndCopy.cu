#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include "CUDAStreamer.h"

__global__ void ConvertAndCopy(const CUDAStreamer::Consumer_element_t* in, cufftReal* out, size_t size)
{

}