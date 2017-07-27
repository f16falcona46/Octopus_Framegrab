#pragma once
#include <string>
#include <thread>
#include <fstream>
#include "BufferQueue.h"

class SaveStreamer
{
public:
	typedef float Consumer_element_t;
	typedef BufferQueue<Consumer_element_t*> Consumer_queue_t;
	SaveStreamer(Consumer_queue_t* in, Consumer_queue_t* out);
	~SaveStreamer();
	Consumer_queue_t* GetConsumerInputQueue() { return m_cons_in; }
	void SetConsumerInputQueue(Consumer_queue_t* in) { m_cons_in = in; }
	Consumer_queue_t* GetConsumerOutputQueue() { return m_cons_out; }
	void SetConsumerOutputQueue(Consumer_queue_t* out) { m_cons_out = out; }
	const std::string& GetFilename();
	void SetFilename(const std::string& filename);
	void Setup();
	void StartStreaming();
	void StopStreaming();

private:
	Consumer_queue_t* m_cons_in;
	Consumer_queue_t* m_cons_out;
	std::string m_filename;
	std::thread m_streaming;
	bool m_streaming;
	std::ofstream m_outfile;
	bool m_setup;
	size_t m_bufsize;

	friend static void StreamFunc(SaveStreamer* streamer);
};

