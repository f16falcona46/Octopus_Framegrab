#include "LoadStreamer.h"
#include <iostream>
#include <algorithm>

LoadStreamer::LoadStreamer() : m_streaming(false), m_setup(false)
{
}


LoadStreamer::~LoadStreamer()
{
	StopStreaming();
	m_input_file.close();
}

void LoadStreamer::Setup()
{
	m_bufcount = m_framewidth * m_frameheight;
	m_bufsize = m_bufcount * sizeof(Producer_element_t);
	m_rdbuf.reset(new Producer_element_t[m_bufcount]);
	m_input_file.open(m_input_filename, std::ios::in | std::ios::binary);
	m_setup = true;
}

void LoadStreamer::StartStreaming()
{
	m_streaming = true;
	m_streamthread = std::thread(StreamFunc, this);
}

void LoadStreamer::StopStreaming()
{
	m_streaming = false;
	if (m_streamthread.joinable()) {
		m_streamthread.join();
	}
}

template <typename T>
void transpose(T* src, T* dst, const int N, const int M) {
	for (int n = 0; n<N*M; n++) {
		int i = n / N;
		int j = n%N;
		dst[n] = src[M*j + i];
	}
}

void LoadStreamer::StreamFunc(LoadStreamer* streamer)
{
	while (streamer->m_streaming && streamer->m_input_file.read((char*)streamer->m_rdbuf.get(), streamer->m_bufsize)) {
		if (streamer->m_prod_in->size() > 0) {
			Producer_element_t* buf = streamer->m_prod_in->front();
			streamer->m_prod_in->pop_front();
			//transpose(streamer->m_rdbuf.get(), buf, streamer->m_framewidth, streamer->m_frameheight);
			std::copy(streamer->m_rdbuf.get(), streamer->m_rdbuf.get() + streamer->m_bufcount, buf);
			std::cout << buf[streamer->m_bufcount - 1] << '\n';
			streamer->m_prod_out->push_back(buf);
		}
	}
}
