#pragma once
#include <string>
#include <thread>
#include <fstream>
#include "BufferQueue.h"

class SaveStreamer
{
public:
	typedef uint16_t Consumer_element_t;
	typedef BufferQueue<Consumer_element_t*> Consumer_queue_t;
	SaveStreamer();
	~SaveStreamer();
	Consumer_queue_t* GetConsumerInputQueue() { return m_cons_in; }
	void SetConsumerInputQueue(Consumer_queue_t* in) { m_cons_in = in; }
	Consumer_queue_t* GetConsumerOutputQueue() { return m_cons_out; }
	void SetConsumerOutputQueue(Consumer_queue_t* out) { m_cons_out = out; }
	void SetBufferSize(size_t bufsize) { m_bufsize = bufsize; }
	const std::string& GetFilename();
	void SetFilename(const std::string& filename);
	void Setup();
	void StartStreaming();
	void StopStreaming();

private:
	Consumer_queue_t* m_cons_in;
	Consumer_queue_t* m_cons_out;
	std::string m_filename;
	std::thread m_streamthread;
	bool m_streaming;
	std::ofstream m_outfile;
	bool m_setup;
	size_t m_bufsize;

	friend void StreamFunc(SaveStreamer* streamer);
};

