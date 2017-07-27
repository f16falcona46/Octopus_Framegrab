#pragma once
#include <cstdint>
#include <thread>
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
	void Setup();
	void StartStreaming();
	void StopStreaming();

private:
	Consumer_queue_t* m_cons_in;
	Consumer_queue_t* m_cons_out;
	Producer_queue_t* m_prod_in;
	Producer_queue_t* m_prod_out;
	std::thread m_streaming;
};

