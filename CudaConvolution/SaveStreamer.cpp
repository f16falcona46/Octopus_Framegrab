#include "SaveStreamer.h"
#include <stdexcept>
#include <iostream>

void SaveStreamer::StreamFunc(SaveStreamer* streamer)
{
	while (streamer->m_streaming) {
		if (streamer->m_cons_in->size() > 0) {
			SaveStreamer::Consumer_element_t* buf = streamer->m_cons_in->front();
			streamer->m_cons_in->pop_front();
			streamer->m_outfile.write((char*)buf, streamer->m_bufsize);
			streamer->m_cons_out->push_back(buf);
		}
	}
}

SaveStreamer::SaveStreamer()
{
}


SaveStreamer::~SaveStreamer()
{
	try {
		StopStreaming();
		m_outfile.close();
	}
	catch (...) {
		std::cout << "Something bad happened in ~SaveStreamer().\n";
	}
}

std::string SaveStreamer::GetFilename()
{
	return m_filename;
}

void SaveStreamer::SetFilename(const std::string& filename)
{
	m_filename = filename;
}

void SaveStreamer::Setup()
{
	m_outfile.close();
	m_outfile.open(m_filename, std::ios::out | std::ios::binary);
	m_setup = true;
}

void SaveStreamer::StartStreaming()
{
	if (!m_setup) throw std::runtime_error("SaveStreamer wasn't set up before calling StartStreaming().");
	m_streaming = true;
	m_streamthread = std::thread(StreamFunc, this);
}

void SaveStreamer::StopStreaming()
{
	m_streaming = false;
	m_streamthread.join();
}
