#pragma once
#include <cstdint>
#include <string>
#include <thread>
#include <SapClassBasic.h>
#include "BufferQueue.h"

class FrameGrabStreamer
{
public:
	typedef uint16_t Producer_element_t;
	typedef BufferQueue<Producer_element_t*> Producer_queue_t;
	FrameGrabStreamer();
	~FrameGrabStreamer();
	Producer_queue_t* GetProducerInputQueue() { return m_prod_in; }
	void SetProducerInputQueue(Producer_queue_t* in) { m_prod_in = in; }
	Producer_queue_t* GetProducerOutputQueue() { return m_prod_out; }
	void SetProducerOutputQueue(Producer_queue_t* out) { m_prod_out = out; }

	void SetServerId(int server_id) { m_server_id = server_id; }
	int GetServerId() { return m_server_id; }
	void SetConfigFile(const std::string& config_file) { m_config_file = config_file; }
	std::string GetConfigFile() { return m_config_file; }
	int GetFrameWidth();
	int GetFrameHeight();

	void Setup();
	void StartStreaming();
	void StopStreaming();

private:
	Producer_queue_t* m_prod_in;
	Producer_queue_t* m_prod_out;
	std::thread m_streaming;
	std::string m_config_file;
	int m_server_id;
	SapLocation m_loc;
	SapAcquisition m_acq;
	SapBufferWithTrash m_buffer;
	SapTransfer m_acq_to_buf;
	bool m_set_up;
	size_t m_buf_size;

	friend void FrameCallback(SapXferCallbackInfo* info);
};

