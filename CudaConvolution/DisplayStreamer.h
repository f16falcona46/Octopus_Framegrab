#pragma once
#include <cstdint>
#include <vector>
#include <Windows.h>
#include "cufft.h"
#include "BufferQueue.h"

class DisplayStreamer
{
public:
	typedef float Consumer_element_t;
	typedef BufferQueue<Consumer_element_t*> Consumer_queue_t;
	DisplayStreamer();
	~DisplayStreamer();
	Consumer_queue_t* GetProducerInputQueue() { return m_prod_in; }
	void SetProducerInputQueue(Consumer_queue_t* in) { m_prod_in = in; }
	Consumer_queue_t* GetProducerOutputQueue() { return m_prod_out; }
	void SetProducerOutputQueue(Consumer_queue_t* out) { m_prod_out = out; }

	void SetLineLength(size_t len) { m_line.resize(len); }

	void Setup();
	void StartStreaming();
	void StopStreaming();

private:
	Consumer_queue_t* m_prod_in;
	Consumer_queue_t* m_prod_out;
	std::thread m_streamthread;
	bool m_streaming;
	bool m_setup;
	HWND m_hwnd;
	std::vector<Consumer_element_t> m_line;

	static void MessageLoop(DisplayStreamer* streamer);
	static LRESULT CALLBACK DisplayStreamerWinProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
};
