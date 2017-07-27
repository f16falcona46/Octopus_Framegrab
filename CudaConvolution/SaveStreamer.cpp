#include "SaveStreamer.h"
#include <stdexcept>

static void StreamFunc(SaveStreamer* streamer)
{
	while (streamer->m_streaming) {
		if (streamer->m_cons_in->size() <= 0) continue;
		SaveStreamer::Consumer_element_t* buf = streamer->m_cons_in->front();
		streamer->m_cons_in->pop_front();
		streamer->m_outfile.write((char*)buf, streamer->m_bufsize);
		streamer->m_cons_out->push_back(buf);
	}
}

SaveStreamer::SaveStreamer(Consumer_queue_t* in, Consumer_queue_t* out)
{
	SetConsumerInputQueue(in);
	SetConsumerOutputQueue(out);
}


SaveStreamer::~SaveStreamer()
{
}

const std::string & SaveStreamer::GetFilename()
{
	return m_filename;
}

void SaveStreamer::SetFilename(const std::string& filename)
{
	m_filename = filename;
}

void SaveStreamer::Setup()
{
	m_outfile.open(m_filename, std::ios::out | std::ios::binary);
	m_setup = true;
}

void SaveStreamer::StartStreaming()
{

}

void SaveStreamer::StopStreaming()
{
}
