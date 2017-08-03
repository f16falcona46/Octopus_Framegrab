#include "ShortSaveStreamer.h"
#include <stdexcept>
#include <iostream>

void ShortSaveStreamer::StreamFunc(ShortSaveStreamer* streamer)
{
	while (streamer->m_streaming) {
		if (streamer->m_cons_in->size() > 0) {
			ShortSaveStreamer::Consumer_element_t* buf = streamer->m_cons_in->front();
			streamer->m_cons_in->pop_front();
			streamer->m_outfile.write((char*)buf, streamer->m_bufsize);
			streamer->m_cons_out->push_back(buf);
		}
	}
}

ShortSaveStreamer::ShortSaveStreamer()
{
}


ShortSaveStreamer::~ShortSaveStreamer()
{
	try {
		StopStreaming();
		m_outfile.close();
	}
	catch (...) {
		std::cout << "Something bad happened in ~ShortSaveStreamer().\n";
	}
}

std::string ShortSaveStreamer::GetFilename()
{
	return m_filename;
}

void ShortSaveStreamer::SetFilename(const std::string& filename)
{
	m_filename = filename;
}

void ShortSaveStreamer::Setup()
{
	m_outfile.close();
	m_outfile.open(m_filename, std::ios::out | std::ios::binary);
	m_setup = true;
}

void ShortSaveStreamer::StartStreaming()
{
	if (!m_setup) throw std::runtime_error("SaveStreamer wasn't set up before calling StartStreaming().");
	m_streaming = true;
	m_streamthread = std::thread(StreamFunc, this);
}

void ShortSaveStreamer::StopStreaming()
{
	m_streaming = false;
	if (m_streamthread.joinable()) {
		m_streamthread.join();
	}
}
