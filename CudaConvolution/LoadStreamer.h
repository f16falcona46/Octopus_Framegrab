#pragma once
#include <cstdint>
#include <string>
#include <thread>
#include "BufferQueue.h"

class LoadStreamer
{
public:
	typedef uint16_t Producer_element_t;
	typedef BufferQueue<Producer_element_t*> Producer_queue_t;
	LoadStreamer();
	~LoadStreamer();
	Producer_queue_t* GetProducerInputQueue() { return m_prod_in; }
	void SetProducerInputQueue(Producer_queue_t* in) { m_prod_in = in; }
	Producer_queue_t* GetProducerOutputQueue() { return m_prod_out; }
	void SetProducerOutputQueue(Producer_queue_t* out) { m_prod_out = out; }

	void SetInputFile(const std::string& input_file) { m_input_file = input_file; }
	std::string GetInputFile() { return m_input_file; }
	void SetFrameWidth(int width) { m_framewidth = width; }
	void SetFrameHeight(int height) { m_frameheight = height; }
	int GetFrameWidth() { return m_framewidth; }
	int GetFrameHeight() { return m_frameheight; }

	void Setup();
	void StartStreaming();
	void StopStreaming();

private:
	Producer_queue_t* m_prod_in;
	Producer_queue_t* m_prod_out;
	std::thread m_streamthread;
	std::string m_input_file;
	bool m_streaming;
	bool m_set_up;
	size_t m_buf_size;
	int m_framewidth;
	int m_frameheight;

	static void StreamFunc(LoadStreamer* streamer);
};

