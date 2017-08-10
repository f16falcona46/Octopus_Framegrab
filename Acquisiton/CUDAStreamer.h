#pragma once
#include <cstdint>
#include <thread>
#include <cufft.h>
#include "BufferQueue.h"

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

	/*
	\brief Initializes cuFFT and allocates CUDA buffers

	This function must be called before StartStreaming() or CopyDCBuffer() is called.

	Upon any change to the stream parameters (e.g., changing the A-line length),
	Setup() must be called again.
	*/
	void Setup();

	/*
	\brief Starts the stream thread and starts streaming

	This function does not block.

	This function must not be called before Setup() is called.
	*/
	void StartStreaming();

	/*
	\brief Stops streaming by stopping the stream thread

	This function blocks until streaming stops. Streaming will stop as soon as the
	buffer currently being processed has been completely processed. However, other
	buffers in the input buffer queue will not be processed.

	This function must not be called before StartStreaming() is called.
	*/
	void StopStreaming();

	/*
	\brief Sets the total number of pixels in each frame

	The number of pixels should be an even multiple of the A-line length.

	\param bufcount	- the total number of pixels in a frame
	*/
	void SetBufferCount(size_t bufcount) { m_bufcount = bufcount; }

	/*
	\brief Sets the A-line length

	\param width	- the length of one A-line, in pixels
	*/
	void SetLineWidth(size_t width) { m_linewidth = width; }

	/*
	\brief Averages and copies DC buffer into the CUDAStreamer

	This function averages all the lines in the DC frame, then copies that
	averaged line into an internal CUDA buffer.

	This function must only be called after Setup() is called.

	\param buf	- a full frame of DC data, in row-major order
	*/
	void CopyDCBuffer(Consumer_element_t* buf);

private:
	Consumer_queue_t* m_cons_in;	//consumer BufferQueue providing data to CUDAStreamer
	Consumer_queue_t* m_cons_out;	//consumer BufferQueue that returns used buffers to the producer
	Producer_queue_t* m_prod_in;	//producer BufferQueue with unused buffers
	Producer_queue_t* m_prod_out;	//producer BufferQueue with processed data from CUDAStreamer
	std::thread m_streamthread;	//thread that runs StreamFunc, which does the streaming
	bool m_streaming;	//if this is false, streamthread will stop
	bool m_buffers_allocated;	//true if CUDA buffers are allocated
	bool m_setup;	//true if Setup() has been called
	cufftHandle m_plan;	//plan for conducting FFT
	uint16_t* m_device_in_buf; //CUDA raw data buffer
	cufftComplex* m_device_conv_in_buf;	//CUDA FFT input buffer (after interpolation, DC subtraction, and conversion to complex)
	cufftComplex* m_device_out_buf;	//CUDA FFT output buffer
	Producer_element_t* m_device_norm_out_buf;	//CUDA buffer for FFT output after norm and log
	Producer_element_t* m_device_contrast_out_buf;	//CUDA buffer for contrast adjusted output
	int* m_device_lerp_index;	//CUDA buffer for linear interpolation indexes (the index of the lower sample point which brackets the target point)
	float* m_device_lerp_fraction;	//CUDA buffer for linear interpolation fractions (how far it is from the upper sample point of the bracket)
	float* m_device_dc_buf;	//CUDA buffer for one line of DC data
	size_t m_bufcount;	//total number of pixels in one frame
	size_t m_linewidth;	//length of one A-line
	size_t m_in_bufsize;	//m_bufcount * sizeof(Consumer_element_t)
	size_t m_out_bufsize;	//m_bufcount * sizeof(Producer_element_t)

	/*
	\brief Frees CUDA buffers and destroys cuFFT plan

	This should only be called if CUDA buffers are actually allocated (check m_buffers_allocated).
	*/
	void DestroyBuffers();

	/*
	\brief Run by the stream thread; performs the streaming

	\param streamer	- CUDAStreamer holding the variables
	*/
	static void CUDAStreamer::StreamFunc(CUDAStreamer* streamer);

	/*
	\brief A simple function to fill a CUDA buffer with a certain value

	Note: defined in CUDAStreamer.cu, not in CUDAStreamer.cpp.

	\param buf	- the CUDA buffer to fill
	\param size	- the number of values to fill (in counts, not in bytes)
	\param value	- the value to fill with
	*/
	static void CUDAStreamer::FillBuffer(cufftComplex* buf, int size, cufftComplex value);

	/*
	\brief Processes raw OCT data into an image.

	Note: defined in CUDAStreamer.cu, not in CUDAStreamer.cpp.

	\param streamer	- CUDAStreamer holding the variables
	*/
	static void DoFFT(CUDAStreamer* streamer);
};